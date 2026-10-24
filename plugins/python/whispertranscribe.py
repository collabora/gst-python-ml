# WhisperTranscribe
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from log.global_logger import GlobalLogger
import backend

CAN_REGISTER_ELEMENT = True
try:
    from backend import GObject
    from base_transcribe import BaseTranscribe
    from engine.whisper_engine import ExportedWhisper, WhisperEngine
    from engine.engine_factory import EngineFactory

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'mp_whispertranscribe' element will not be available. Error: {e}"
    )


class WhisperTranscribe(BaseTranscribe):
    __gstmetadata__ = (
        "WhisperTranscribe",
        "Text Output",
        "Python element that transcribes audio with Whisper",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    def __init__(self):
        super().__init__()
        self._beam_size = 5
        self.model_name = "medium"
        self.task_engine_name = self.mgr.engine_name = "mp_whispertranscribe_engine"
        EngineFactory.register(self.mgr.engine_name, WhisperEngine)

    def export_model(self, model_name):
        return ExportedWhisper(model_name)

    @GObject.Property(type=int, default=5, minimum=1, maximum=8)
    def beam_size(self):
        return self._beam_size

    @beam_size.setter
    def beam_size(self, value):
        self._beam_size = value

    def do_transcribe(self, audio_data, task):
        return self.task_engine.transcribe(
            audio_data, self.language, task, self._beam_size, self.initial_prompt
        )


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element(
        "mp_whispertranscribe", WhisperTranscribe
    )
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'mp_whispertranscribe' element will not be registered because base_transcribe module is missing."
    )
