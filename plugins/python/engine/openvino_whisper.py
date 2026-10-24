# OpenVinoWhisper
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import functools

from .hub_causal_lm import export_once, hub_cache_name
from .openvino_llm import OPENVINO_LLM_CACHE

EXPORTED_MODEL_NAME = "openvino_encoder_model.xml"
EXPORT_TASK = "automatic-speech-recognition-with-past"


# optimum 2.3 stores a partial that python 3.14 binds as a method
def unbind_normalized_config(config_class):
    normalized = vars(config_class).get("NORMALIZED_CONFIG_CLASS")
    if isinstance(normalized, functools.partial):
        config_class.NORMALIZED_CONFIG_CLASS = staticmethod(normalized)


def export_model(model_name, output):
    from optimum.exporters.openvino import main_export
    from optimum.exporters.openvino.model_configs import WhisperOpenVINOConfig

    unbind_normalized_config(WhisperOpenVINOConfig)
    main_export(model_name, output, task=EXPORT_TASK, convert_tokenizer=True)


def exported_model_path(model_name):
    path = OPENVINO_LLM_CACHE / hub_cache_name(model_name)
    return export_once(
        path, EXPORTED_MODEL_NAME, lambda output: export_model(model_name, output)
    )


class OpenVinoWhisper:
    def __init__(self, model_name, device):
        import openvino_genai

        self.pipeline = openvino_genai.WhisperPipeline(
            str(exported_model_path(model_name)), device
        )

    def transcribe(self, audio, language, task, beam_size, initial_prompt):
        config = self.pipeline.get_generation_config()
        config.language = f"<|{language}|>" if language else None
        config.task = task
        config.num_beams = beam_size
        config.initial_prompt = initial_prompt or None
        return [text.strip() for text in self.pipeline.generate(audio, config).texts]
