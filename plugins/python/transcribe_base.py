# TranscribeBase
# Copyright (C) 2024-2025 Collabora Ltd.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Library General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Library General Public License for more details.
#
# You should have received a copy of the GNU Library General Public
# License along with this library; if not, write to the
# Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
# Boston, MA 02110-1301, USA.

import collections
import sys
from abc import abstractmethod

import numpy as np
from pysilero_vad import SileroVoiceActivityDetector
import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
gi.require_version("GObject", "2.0")
from gi.repository import Gst, GObject, GstBase  # noqa: E402

from aggregator_base import AggregatorBase  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

STT_SAMPLE_RATE = 16000  # Target sample rate for processing

# Initialize VAD
vad = SileroVoiceActivityDetector()
vad_chunk_size = vad.chunk_samples()

ICAPS = Gst.Caps(
    Gst.Structure(
        "audio/x-raw",
        format="S16LE",
        layout="interleaved",
        rate=STT_SAMPLE_RATE,
        channels=1,
    )
)

OCAPS = Gst.Caps(Gst.Structure("text/x-raw", format="utf8"))


class TranscribeBase(AggregatorBase):
    __gstmetadata__ = (
        "TranscribeBase",
        "Text Output",
        "Python element that transcribes audio with Whisper",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    __gsttemplates__ = (
        Gst.PadTemplate.new_with_gtype(
            "sink",
            Gst.PadDirection.SINK,
            Gst.PadPresence.REQUEST,
            ICAPS,
            GstBase.AggregatorPad.__gtype__,
        ),
        Gst.PadTemplate.new_with_gtype(
            "src",
            Gst.PadDirection.SRC,
            Gst.PadPresence.ALWAYS,
            OCAPS,
            GstBase.AggregatorPad.__gtype__,
        ),
    )

    initial_prompt = GObject.Property(
        type=str,
        default="",
        nick="Initial Prompt",
        blurb="Comma-separated list of initial prompt keywords for transcription.",
    )

    translate = GObject.Property(
        type=bool,
        default=False,
        nick="Translate",
        blurb="Translate audio to English.",
    )

    language = GObject.Property(
        type=str,
        default="en",
        nick="Language",
        blurb="Two-character code for the language to transcribe.",
    )

    streaming = GObject.Property(
        type=bool,
        default=False,
        nick="Streaming",
        blurb="Enable or disable streaming mode. If disabled, process in batch mode.",
    )

    def __init__(self):
        super().__init__()

        self.clip_buffer = collections.deque()
        self.active_clip = False
        self.silence_counter = 0
        chunk_duration_ms = (vad_chunk_size / STT_SAMPLE_RATE) * 1000
        silence_ms = 300
        self.clip_silence_trigger_counter = int(silence_ms / chunk_duration_ms)
        self.old_device = None

    def do_get_property(self, prop):
        if prop.name == "initial_prompt":
            return self.initial_prompt
        elif prop.name == "translate":
            return self.translate
        elif prop.name == "language":
            return self.language
        elif prop.name == "streaming":
            return self.streaming
        else:
            raise AttributeError(f"Unknown property: {prop.name}")

    def do_set_property(self, prop, value):
        if prop.name == "initial_prompt":
            self.initial_prompt = value
        elif prop.name == "translate":
            self.translate = value
        elif prop.name == "language":
            self.language = value
        elif prop.name == "streaming":
            self.streaming = value
        else:
            raise AttributeError(f"Unknown property: {prop.name}")

    @abstractmethod
    def do_load_model(self):
        pass

    @abstractmethod
    def do_transcribe(self, audio_data, task):
        pass

    def do_process_text(self, transcript):
        # Encode the transcript as UTF-8
        text_bytes = transcript.encode("utf-8")
        encoded_size = len(text_bytes)

        # Create a new buffer for output and write the transcription
        outbuf = Gst.Buffer.new_allocate(None, encoded_size, None)
        outbuf.fill(0, text_bytes)

        return outbuf

    def do_process(self, buf):
        self.push_segment_if_needed()
        """Process audio data from the input buffers using VAD and Whisper."""
        audio_collected = False

        try:
            if self.device != self.old_device:
                self.do_load_model()

            # Map the buffer to access the audio data
            success, map_info = buf.map(Gst.MapFlags.READ)
            if not success:
                self.logger.error("Failed to map input buffer")
                buf.unmap(map_info)
                return Gst.FlowReturn.OK

            # Convert buffer to numpy array (int16)
            audio_data = np.frombuffer(map_info.data, dtype=np.int16)
            audio_collected = True

            if len(audio_data) < vad_chunk_size:
                self.logger.warning("Insufficient audio data for processing")
                buf.unmap(map_info)
                return Gst.FlowReturn.OK

            # Process audio data with VAD (Voice Activity Detection)
            while len(audio_data) >= vad_chunk_size:
                vad_chunk = audio_data[:vad_chunk_size]
                audio_data = audio_data[vad_chunk_size:]

                vad_confidence = vad.process_chunk(vad_chunk.tobytes())
                if vad_confidence >= 0.7:
                    if self.streaming:
                        transcript = self._transcribe_audio(vad_chunk)
                        if transcript is None:
                            self.logger.warning("Empty transcript")
                            buf.unmap(map_info)
                            return Gst.FlowReturn.ERROR

                        self._process_and_send(transcript, buf)
                    else:
                        # VAD detects voice activity, add to buffer
                        self.active_clip = True
                        self.silence_counter = 0
                        self.clip_buffer.extend(vad_chunk)
                else:
                    # Increment silence counter when no voice is detected
                    self.silence_counter += 1

                    # If silence is detected for too long, end the current segment
                    if (
                        self.active_clip
                        and self.silence_counter > self.clip_silence_trigger_counter
                    ):
                        self.active_clip = False
                        if not self.streaming:
                            # Perform transcription in batch mode
                            transcript = self._transcribe_audio(self.clip_buffer)
                            if transcript is None:
                                self.logger.warning("Empty transcript")
                                buf.unmap(map_info)
                                return Gst.FlowReturn.ERROR

                            self._process_and_send(transcript, buf)
                            self.clip_buffer.clear()  # Clear the buffer for the next speech
            buf.unmap(map_info)
        except Exception as e:
            self.logger.error(f"Error during buffer processing: {e}")

        if not audio_collected:
            self.logger.warning("No audio data collected from sink pads.")
            return Gst.FlowReturn.ERROR

        return Gst.FlowReturn.OK

    def _process_and_send(self, transcript, inbuf):
        outbuf = self.do_process_text(transcript)
        if outbuf is None:
            return

        # Set PTS and duration from the input buffer
        outbuf.pts = inbuf.pts
        outbuf.duration = inbuf.duration

        # Push the transcription downstream
        self.finish_buffer(outbuf)

    def _transcribe_audio(self, chunk):
        """
        Transcribes the buffered audio data
        and returns the transcript for streaming.
        """
        try:
            # Get the current audio data from the buffer for streaming transcription
            audio_data = np.array(chunk).astype(np.float32) / 32768.0
            task = "translate" if self.translate else "transcribe"
            result = self.do_transcribe(audio_data, task)
            # Combine all segments into a single transcript
            transcript = " ".join([seg.text.strip() for seg in list(result)])
            self.logger.info(f"transcription: {transcript}")
            return transcript

        except Exception as e:
            self.logger.error(f"Error during streaming transcription: {e}")
            return ""
