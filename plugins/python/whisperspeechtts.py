# WhisperSpeechTTS
# Copyright (C) 2024 Collabora Ltd.
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

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
from gi.repository import Gst, GObject, GstBase  # noqa: E402

CAN_REGISTER_ELEMENT = True
try:
    import numpy as np
    from whisperspeech.pipeline import Pipeline
    from gst_analytics_tts import GstAnalyticsTTS
except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    Gst.warning(f"The 'whisperspeechtts' element will not be available. Error: {e}")

TTS_SAMPLE_RATE = 24000

model_ref = "collabora/whisperspeech:s2a-q4-base-en+pl.model"

OCAPS = Gst.Caps(
    Gst.Structure(
        "audio/x-raw",
        format="S16LE",
        layout="interleaved",
        rate=TTS_SAMPLE_RATE,
        channels=1,
    )
)


class WhisperSpeechTTS(GstAnalyticsTTS):
    __gstmetadata__ = (
        "WhisperSpeechTTS",
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
        Gst.info(f"Initializing WhisperSpeech TTS model on device: {self.device}")
        try:
            self.set_model(
                Pipeline(s2a_ref=model_ref, device=self.device, torch_compile=True)
            )
            if self.get_model() is not None:
                Gst.info(
                    f"WhisperSpeech Pipeline initialized successfully: {self.get_model()}"
                )
            else:
                Gst.error("Failed to create WhisperSpeech model")
        except Exception as e:
            Gst.error(f"Exception during model initialization: {e}")

    def do_generate_speech(self, transcript):
        audio_tensor = self.get_model().generate(transcript, lang=self.language)
        audio_np = (audio_tensor.cpu().numpy() * 32767).astype(
            np.int16
        )  # Convert tensor to numpy array, scale, and cast to int16
        if len(audio_np.shape) == 1:  # Check if the numpy array is 1D
            audio_np = np.expand_dims(
                audio_np, axis=0
            )  # Add a new dimension to make it 2D
        else:
            audio_np = audio_np.T  # Transpose the numpy array if it's not 1D
        return audio_np

    def do_get_sample_rate(self):
        return TTS_SAMPLE_RATE


if CAN_REGISTER_ELEMENT:
    GObject.type_register(WhisperSpeechTTS)
    __gstelementfactory__ = ("whisperspeechtts", Gst.Rank.NONE, WhisperSpeechTTS)
else:
    Gst.warning(
        "The 'whisperspeechtts' element will not be registered because required modules were missing."
    )
