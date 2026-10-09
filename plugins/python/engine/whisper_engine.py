# WhisperEngine
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

from .hub_causal_lm import HUB_NAME_SEPARATOR
from .ml_engine import MLEngine

WHISPER_SAMPLE_RATE = 16000
OPENAI_WHISPER_PREFIX = "openai/whisper-"


# faster-whisper takes a size name, the other engines the hub checkpoint
def hub_whisper_name(model_name):
    if HUB_NAME_SEPARATOR in model_name:
        return model_name
    return f"{OPENAI_WHISPER_PREFIX}{model_name}"


class ExportedWhisper:
    def __init__(self, model_name):
        self.engine = None
        self.path = hub_whisper_name(model_name)

    def transcribe(self, audio, language, task, beam_size, initial_prompt):
        return self.engine.model.transcribe(
            audio, language, task, beam_size, initial_prompt
        )


class WhisperEngine(MLEngine):
    def do_load_model(self, model_name, **kwargs):
        from faster_whisper import WhisperModel

        if not model_name:
            return
        compute_type = "float16" if self.device.startswith("cuda") else "int8"
        self.logger.info(
            f"Loading Whisper model on device: {self.device} with compute_type: {compute_type}"
        )
        self.model = WhisperModel(
            model_name, device=self.device, compute_type=compute_type
        )

    def transcribe(self, audio, language, task, beam_size, initial_prompt):
        segments, _ = self.model.transcribe(
            audio,
            language=language,
            task=task,
            beam_size=beam_size,
            initial_prompt=initial_prompt,
        )
        return [segment.text.strip() for segment in segments]
