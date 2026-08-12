# CoquiTTS
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

from log.global_logger import GlobalLogger
import backend

CAN_REGISTER_ELEMENT = True
try:
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    from gi.repository import Gst, GstBase  # noqa: E402
    from base_tts import BaseTts
except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'pyml_coquitts' element will not be available. Error: {e}"
    )

TTS_SAMPLE_RATE = 22050


class CoquiTTS(BaseTts):
    __gstmetadata__ = (
        "CoquiTTS",
        "Aggregator",
        "Converts text to audio",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    # the rate has to match TTS_SAMPLE_RATE, which the element reports downstream
    OUTPUT_CAPS = "audio/x-raw,format=S16LE,layout=interleaved,rate=22050,channels=1"

    # Building a Gst object needs Gst.init, which only the gst backend calls.
    if backend.BACKEND == "gst":
        __gsttemplates__ = (
            Gst.PadTemplate.new_with_gtype(
                "src",
                Gst.PadDirection.SRC,
                Gst.PadPresence.ALWAYS,
                Gst.Caps.from_string(OUTPUT_CAPS),
                GstBase.AggregatorPad.__gtype__,
            ),
        )

    def do_load_model(self):
        from TTS.api import TTS

        self.logger.info(f"Initializing Coqui TTS model on device: {self.device}")
        self.set_model(
            TTS(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                progress_bar=False,
            )
        )
        self.get_model().to(self.device)

    def do_generate_speech(self, transcript):
        audio_output = self.get_model().tts(
            text=transcript, speaker=self.speaker, language=self.language
        )
        return audio_output

    def do_get_sample_rate(self):
        return TTS_SAMPLE_RATE


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element("pyml_coquitts", CoquiTTS)
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'pyml_coquitts' element will not be registered because required modules are missing."
    )
