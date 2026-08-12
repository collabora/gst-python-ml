# BaseTranscribe
# Copyright (C) 2024-2026 Collabora Ltd.
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

import backend
from backend import GObject
from base_aggregator import BaseAggregator

if backend.BACKEND == "gst":
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    from gi.repository import Gst, GstBase  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

STT_SAMPLE_RATE = 16000  # Target sample rate for processing


class BaseTranscribe(BaseAggregator):
    __gstmetadata__ = (
        "BaseTranscribe",
        "Text Output",
        "Python element that transcribes audio with Whisper",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    # the caps each pad negotiates, stated once for both backends. The rate has
    # to match STT_SAMPLE_RATE, which the VAD chunking works from.
    INPUT_CAPS = "audio/x-raw,format=S16LE,layout=interleaved,rate=16000,channels=1"
    OUTPUT_CAPS = "text/x-raw,format=utf8"

    # Building a Gst object needs Gst.init, which only the gst backend calls.
    if backend.BACKEND == "gst":
        __gsttemplates__ = (
            Gst.PadTemplate.new_with_gtype(
                "sink",
                Gst.PadDirection.SINK,
                Gst.PadPresence.REQUEST,
                Gst.Caps.from_string(INPUT_CAPS),
                GstBase.AggregatorPad.__gtype__,
            ),
            Gst.PadTemplate.new_with_gtype(
                "src",
                Gst.PadDirection.SRC,
                Gst.PadPresence.ALWAYS,
                Gst.Caps.from_string(OUTPUT_CAPS),
                GstBase.AggregatorPad.__gtype__,
            ),
        )

    def __init__(self):
        super().__init__()

        from pysilero_vad import SileroVoiceActivityDetector

        self._vad = SileroVoiceActivityDetector()
        self._vad_chunk_size = self._vad.chunk_samples()

        self.clip_buffer = collections.deque()
        self.active_clip = False
        self.silence_counter = 0
        chunk_duration_ms = (self._vad_chunk_size / STT_SAMPLE_RATE) * 1000
        silence_ms = 300
        self.clip_silence_trigger_counter = int(silence_ms / chunk_duration_ms)
        self.__initial_prompt = ""
        self.__translate = False
        self.__language = "en"
        self.__streaming = False

    @GObject.Property(type=str, default="")
    def initial_prompt(self):
        "Initial Prompt"
        return self.__initial_prompt

    @initial_prompt.setter
    def initial_prompt(self, value):
        self.__initial_prompt = value

    @GObject.Property(type=bool, default=False)
    def translate(self):
        "toggle translation functionality"
        return self.__translate

    @translate.setter
    def translate(self, value):
        self.__translate = value

    @GObject.Property(type=str, default="en")
    def language(self):
        "two character language code for language to transcribe from"
        return self.__language

    @language.setter
    def language(self, value):
        self.__language = value

    @GObject.Property(type=bool, default=False)
    def streaming(self):
        "toggle streaming"
        return self.__streaming

    @streaming.setter
    def streaming(self, value):
        self.__streaming = value

    @abstractmethod
    def do_transcribe(self, audio_data, task):
        pass

    def do_process_text(self, transcript):
        """The payload one transcript becomes. `None` sends nothing."""
        return transcript.encode("utf-8")

    def process_payload(self, payload: bytes) -> list[bytes]:
        """Runs VAD over the audio, transcribing each clip that ends.

        Speech is accumulated across buffers, so most buffers produce nothing.
        """
        import numpy as np

        audio_data = np.frombuffer(payload, dtype=np.int16)

        if len(audio_data) < self._vad_chunk_size:
            self.logger.warning("Insufficient audio data for processing")
            return []

        payloads = []
        while len(audio_data) >= self._vad_chunk_size:
            vad_chunk = audio_data[: self._vad_chunk_size]
            audio_data = audio_data[self._vad_chunk_size :]

            vad_confidence = self._vad.process_chunk(vad_chunk.tobytes())
            if vad_confidence >= 0.7:
                if self.streaming:
                    self._collect_transcript(payloads, vad_chunk)
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
                        self._collect_transcript(payloads, self.clip_buffer)
                        self.clip_buffer.clear()  # Clear the buffer for the next speech

        return payloads

    def _collect_transcript(self, payloads, chunk):
        transcript = self._transcribe_audio(chunk)
        if transcript is None:
            self.logger.warning("Empty transcript")
            return

        payload = self.do_process_text(transcript)
        if payload is not None:
            payloads.append(payload)

    def _transcribe_audio(self, chunk):
        """
        Transcribes the buffered audio data
        and returns the transcript for streaming.
        """
        import numpy as np

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
