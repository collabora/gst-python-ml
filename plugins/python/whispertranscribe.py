# WhisperTranscribe
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
    from backend import GObject
    from base_transcribe import BaseTranscribe
    from engine.whisper_engine import WhisperEngine
    from engine.engine_factory import EngineFactory

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'pyml_whispertranscribe' element will not be available. Error: {e}"
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
        self.model_name = "medium"
        # set engine name directly since property is read only
        self.mgr.engine_name = "pyml_whispertranscribe_engine"
        EngineFactory.register(self.mgr.engine_name, WhisperEngine)

    # make engine_name read only
    @GObject.Property(type=str)
    def engine_name(self):
        """Machine Learning Engine (read-only in this class)."""
        return self.mgr.engine_name

    @engine_name.setter
    def engine_name(self, value):
        raise ValueError(
            "The 'engine_name' property cannot be set in this derived class."
        )

    def do_transcribe(self, audio_data, task):
        result, _ = self.get_model().transcribe(
            audio_data,
            language=self.language,
            task=task,
            initial_prompt=self.initial_prompt,
        )
        return result


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element(
        "pyml_whispertranscribe", WhisperTranscribe
    )
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'pyml_whispertranscribe' element will not be registered because base_transcribe module is missing."
    )
