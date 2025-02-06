# CoquiTTS
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

from global_logger import GlobalLogger

CAN_REGISTER_ELEMENT = True
try:
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    from gi.repository import Gst, GObject, GstBase  # noqa: E402
    from tts_base import TtsBase
    from TTS.api import TTS
except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'pyml_coquitts' element will not be available. Error: {e}"
    )

TTS_SAMPLE_RATE = 22050

OCAPS = Gst.Caps(
    Gst.Structure(
        "audio/x-raw",
        format="S16LE",
        layout="interleaved",
        rate=TTS_SAMPLE_RATE,
        channels=1,
    )
)


class CoquiTTS(TtsBase):
    __gstmetadata__ = (
        "CoquiTTS",
        "Aggregator",
        "Converts text to audio",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    __gsttemplates__ = (
        Gst.PadTemplate.new_with_gtype(
            "src",
            Gst.PadDirection.SRC,
            Gst.PadPresence.ALWAYS,
            OCAPS,
            GstBase.AggregatorPad.__gtype__,
        ),
    )

    def do_load_model(self):
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


if CAN_REGISTER_ELEMENT:
    GObject.type_register(CoquiTTS)
    __gstelementfactory__ = ("pyml_coquitts", Gst.Rank.NONE, CoquiTTS)
else:
    GlobalLogger().warning(
        "The 'pyml_coquitts' element will not be registered because required modules are missing."
    )
