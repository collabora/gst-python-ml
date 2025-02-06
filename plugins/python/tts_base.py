# TtsBase
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

from abc import abstractmethod
import io
import asyncio
import soundfile as sf
import gi

from aggregator_base import AggregatorBase

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
gi.require_version("GstAudio", "1.0")
from gi.repository import Gst, GObject, GstBase, GstAudio  # noqa: E402

ICAPS = Gst.Caps(Gst.Structure("text/x-raw", format="utf8"))


class TtsBase(AggregatorBase):
    __gstmetadata__ = (
        "TtsBase",
        "Aggregator",
        "Parent TTS class",
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
    )

    language = GObject.Property(
        type=str,
        default="en",
        nick="Language",
        blurb="Two-character code for the language to be used by TTS model.",
    )
    streaming = GObject.Property(
        type=bool,
        default=False,
        nick="Streaming",
        blurb="Enable streaming mode for real-time audio generation.",
    )
    speaker = GObject.Property(
        type=str,
        default="Andrew Chipper",
        nick="Spekar ID",
        blurb="Speaker for TTS model",
    )

    def __init__(self):
        super().__init__()
        self.segment_pushed = False
        self.device = "cpu"
        self.streaming_enabled = False

    @abstractmethod
    def do_load_model(self):
        pass

    @abstractmethod
    def do_generate_speech(self, transcript):
        pass

    @abstractmethod
    def do_get_sample_rate(self):
        pass

    def set_property(self, property_name, value):
        if property_name == "speaker":
            self.speaker = value
        elif property_name == "language":
            self.language = value
        elif property_name == "streaming":
            self.streaming_enabled = value
            self.logger.info(f"Streaming mode {'enabled' if value else 'disabled'}")
        else:
            super().set_property(property_name, value)

    def get_property(self, property_name):
        if property_name == "speaker":
            return self.speaker
        elif property_name == "language":
            return self.language
        elif property_name == "streaming":
            return self.streaming_enabled
        else:
            return super().get_property(property_name)

    def do_set_caps(self, in_caps, out_caps):
        self.audio_info = GstAudio.AudioInfo()
        self.audio_info.set_format(
            GstAudio.AudioFormat.S16LE, self.do_get_sample_rate(), 1, None
        )
        return True

    def do_process(self, buf):
        try:
            success, map_info = buf.map(Gst.MapFlags.READ)
            if not success:
                self.logger.error("Failed to map input buffer")
                return Gst.FlowReturn.ERROR

            byte_data = bytes(map_info.data)
            if not byte_data:
                buf.unmap(map_info)
                return Gst.FlowReturn.OK

            try:
                byte_data = byte_data.decode("utf-8", errors="replace")
            except Exception as e:
                self.logger.error(f"Error decoding text data: {e}")
                buf.unmap(map_info)
                return Gst.FlowReturn.ERROR

            self.logger.info(f"TTS: received text: {byte_data}")

            if self.streaming_enabled:
                self.convert_text_to_audio_streaming_async(byte_data)
            else:
                self.convert_text_to_audio_async(byte_data)

            buf.unmap(map_info)

        except Exception as e:
            self.logger.error(f"Error processing text buffer: {e}")
            return Gst.FlowReturn.ERROR

    async def process_transcript(self, transcript):
        try:
            if self.get_model() is None:
                self.do_load_model()

            tts_output = self.do_generate_speech(transcript)
            with io.BytesIO() as buffer:
                sf.write(
                    buffer,
                    tts_output,
                    samplerate=self.do_get_sample_rate(),
                    format="WAV",
                )
                buffer.seek(0)
                audio_bytes, sr = sf.read(buffer, dtype="int16")

            if sr != self.do_get_sample_rate():
                raise ValueError("Sample rate mismatch in audio processing")

            self.push_audio_to_pipeline(audio_bytes)
        except Exception as e:
            self.logger.error(f"Error processing TTS: {e}")

    def convert_text_to_audio_async(self, text):
        asyncio.run(self.process_transcript(text))

    def convert_text_to_audio_streaming_async(self, text):
        """Converts the text in smaller chunks for streaming."""
        chunks = self.split_text_into_chunks(text, 20)
        for chunk in chunks:
            asyncio.run(self.process_transcript(chunk))

    def split_text_into_chunks(self, text, max_length=50):
        """Splits text into smaller chunks for streaming."""
        return [text[i : i + max_length] for i in range(0, len(text), max_length)]

    def push_audio_to_pipeline(self, audio_data):
        try:
            duration = len(audio_data) / self.do_get_sample_rate() * Gst.SECOND
            buffer = Gst.Buffer.new_wrapped(audio_data.tobytes())

            buffer.pts = Gst.CLOCK_TIME_NONE
            buffer.duration = duration

            ret = self.srcpad.push(buffer)
            if ret != Gst.FlowReturn.OK:
                raise RuntimeError(f"Error pushing audio to pipeline: {ret}")

            self.logger.info("TTS: audio generated and pushed downstream successfully.")

        except Exception as e:
            self.logger.error(f"Error pushing audio to pipeline: {e}")
