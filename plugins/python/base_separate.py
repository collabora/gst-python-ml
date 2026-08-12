# BaseSeparate
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

import traceback
import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
gi.require_version("GObject", "2.0")
from gi.repository import Gst, GstBase  # noqa: E402

import backend  # noqa: E402
from backend import GObject  # noqa: E402
from base_aggregator import BaseAggregator  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


class BaseSeparate(BaseAggregator):
    __gstmetadata__ = (
        "BaseSeparate",
        "Audio Output",
        "Python element that separates audio sources",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    SAMPLE_RATE = 44100  # native sample rate of Demucs

    # the caps each pad negotiates, stated once for both backends. The rate has
    # to match SAMPLE_RATE, which the chunking works from.
    INPUT_CAPS = "audio/x-raw,format=S16LE,layout=interleaved,rate=44100,channels=1"
    OUTPUT_CAPS = "audio/x-raw,format=S16LE,layout=interleaved,rate=44100,channels=1"

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

        self.clip_buffer = collections.deque()
        self.__streaming = False
        self.__stem = "vocals"

    @GObject.Property(type=bool, default=False)
    def streaming(self):
        "toggle streaming"
        return self.__streaming

    @streaming.setter
    def streaming(self, value):
        self.__streaming = value

    @GObject.Property(type=str, default="vocals")
    def stem(self):
        "stem to output (e.g., vocals, drums, bass, other)"
        return self.__stem

    @stem.setter
    def stem(self, value):
        self.__stem = value

    @abstractmethod
    def do_separate(self, audio_data):
        pass

    def do_process_audio(self, audio_data):
        # audio_data is np.int16 array
        return audio_data.tobytes()

    def process_payload(self, payload: bytes) -> list[bytes]:
        """Separates the requested stem, one payload per full chunk of audio.

        Audio is accumulated until there is a whole chunk to separate, so a
        buffer produces anywhere from zero to several payloads.
        """
        import numpy as np

        audio_data = np.frombuffer(payload, dtype=np.int16)
        self.clip_buffer.extend(audio_data)

        chunk_duration = 1.0 if self.streaming else 10.0
        chunk_size = int(self.SAMPLE_RATE * chunk_duration)

        payloads = []
        while len(self.clip_buffer) >= chunk_size:
            chunk = np.fromiter(
                (self.clip_buffer.popleft() for _ in range(chunk_size)),
                dtype=np.int16,
            )
            separated = self._separate_audio(chunk)
            if separated is None:
                self.logger.warning("Empty separated audio")
                return payloads

            payloads.append(self.do_process_audio(separated))

        return payloads

    def _separate_audio(self, chunk):
        """
        Separates the buffered audio data
        and returns the separated stem as np.int16.
        """
        import numpy as np

        try:
            # Get the current audio data from the buffer for separation
            audio_data = chunk.astype(np.float32) / 32768.0
            result = self.do_separate(audio_data)
            # Convert back to int16
            separated = np.clip(result * 32768, -32768, 32767).astype(np.int16)
            self.logger.info(f"Separated audio length: {len(separated)}")
            return separated

        except Exception as e:
            self.logger.error(f"Error during separation: {e}\n{traceback.format_exc()}")
            return None
