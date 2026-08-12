# BaseTts
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

from abc import abstractmethod
import io
import asyncio

import backend
from backend import GObject
from base_aggregator import BaseAggregator

if backend.BACKEND == "gst":
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    gi.require_version("GstAudio", "1.0")
    from gi.repository import Gst, GstBase, GstAudio  # noqa: E402

BYTES_PER_SAMPLE = 2  # S16LE, the format every subclass produces
NANOSECONDS_PER_SECOND = 1_000_000_000


class BaseTts(BaseAggregator):
    __gstmetadata__ = (
        "BaseTts",
        "Aggregator",
        "Parent TTS class",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    PUSH_FROM_SRC_PAD = True

    # the sink pad caps, stated once for both backends; each subclass declares
    # the audio it produces
    INPUT_CAPS = "text/x-raw,format=utf8"

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
        )

    language = GObject.Property(
        type=str,
        default="en",
        nick="Language",
        blurb="Two-character code for the language to be used by TTS model.",
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
        self.__streaming = False

    @GObject.Property(type=bool, default=False)
    def streaming(self):
        "Enable streaming mode for real-time audio generation."
        return self.__streaming

    @streaming.setter
    def streaming(self, value):
        self.__streaming = value
        self.logger.info(f"Streaming mode {'enabled' if value else 'disabled'}")

    @abstractmethod
    def do_load_model(self):
        pass

    @abstractmethod
    def do_generate_speech(self, transcript):
        pass

    @abstractmethod
    def do_get_sample_rate(self):
        pass

    def do_set_caps(self, in_caps, out_caps):
        self.audio_info = GstAudio.AudioInfo()
        self.audio_info.set_format(
            GstAudio.AudioFormat.S16LE, self.do_get_sample_rate(), 1, None
        )
        return True

    def process_payload(self, payload: bytes) -> list[bytes]:
        """Speaks the input text, one payload per stretch of generated audio."""
        if not payload:
            return []

        text = payload.decode("utf-8", errors="replace")
        self.logger.info(f"TTS: received text: {text}")

        chunks = self.split_text_into_chunks(text, 20) if self.streaming else [text]
        payloads = []
        for chunk in chunks:
            audio = asyncio.run(self.process_transcript(chunk))
            if audio is not None:
                payloads.append(audio.tobytes())

        return payloads

    def payload_duration_ns(self, payload_size):
        samples = payload_size // BYTES_PER_SAMPLE
        return int(samples / self.do_get_sample_rate() * NANOSECONDS_PER_SECOND)

    async def process_transcript(self, transcript):
        import soundfile as sf

        try:
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

            return audio_bytes
        except Exception as e:
            self.logger.error(f"Error processing TTS: {e}")
            return None

    def split_text_into_chunks(self, text, max_length=50):
        """Splits text into smaller chunks for streaming."""
        return [text[i : i + max_length] for i in range(0, len(text), max_length)]
