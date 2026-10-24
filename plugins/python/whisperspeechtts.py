# WhisperSpeechTTS
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from log.global_logger import GlobalLogger
import backend

CAN_REGISTER_ELEMENT = True
try:
    from base_tts import BaseTts
except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'mp_whisperspeechtts' element will not be available. Error: {e}"
    )

if backend.BACKEND == "gst":
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    from gi.repository import Gst, GstBase  # noqa: E402

TTS_SAMPLE_RATE = 24000

model_ref = "collabora/whisperspeech:s2a-q4-base-en+pl.model"


# a new tts backend is a BaseTts subclass with OUTPUT_CAPS, do_load_model and do_generate_speech
class WhisperSpeechTTS(BaseTts):
    __gstmetadata__ = (
        "WhisperSpeechTTS",
        "Aggregator",
        "Converts text to audio",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    # the rate has to match TTS_SAMPLE_RATE, which the element reports downstream
    OUTPUT_CAPS = "audio/x-raw,format=S16LE,layout=interleaved,rate=24000,channels=1"

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
        # the whisperspeech package moves its models with .cuda()
        if not self.device.startswith("cuda"):
            raise ValueError("whisperspeech runs on cuda only")

        from whisperspeech.pipeline import Pipeline

        self.logger.info(
            f"Initializing WhisperSpeech TTS model on device: {self.device}"
        )
        self.set_model(Pipeline(s2a_ref=model_ref))

    def do_generate_speech(self, transcript):
        audio_tensor = self.get_model().generate(transcript, lang=self.language)
        return audio_tensor.float().cpu().numpy().reshape(-1)

    def do_get_sample_rate(self):
        return TTS_SAMPLE_RATE


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element(
        "mp_whisperspeechtts", WhisperSpeechTTS
    )
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'mp_whisperspeechtts' element will not be registered because required modules were missing."
    )
