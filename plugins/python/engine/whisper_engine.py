# WhisperEngine
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

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
