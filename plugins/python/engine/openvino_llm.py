# OpenVinoLLM
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from .hub_causal_lm import chat_prompt, export_once, hub_cache_name
from .ml_engine import MODEL_CACHE

OPENVINO_LLM_CACHE = MODEL_CACHE / "openvino-genai"
EXPORTED_MODEL_NAME = "openvino_model.xml"
EXPORT_TASK = "text-generation-with-past"
WEIGHT_BITS = 4


def export_model(model_name, output):
    from optimum.exporters.openvino import main_export
    from optimum.intel import OVConfig, OVWeightQuantizationConfig

    main_export(
        model_name,
        output,
        task=EXPORT_TASK,
        ov_config=OVConfig(
            quantization_config=OVWeightQuantizationConfig(bits=WEIGHT_BITS)
        ),
        convert_tokenizer=True,
    )


def exported_model_path(model_name):
    path = OPENVINO_LLM_CACHE / f"{hub_cache_name(model_name)}-int{WEIGHT_BITS}"
    return export_once(
        path, EXPORTED_MODEL_NAME, lambda output: export_model(model_name, output)
    )


class OpenVinoLLM:
    def __init__(self, model_name, device):
        import openvino_genai
        from transformers import AutoTokenizer

        self.pipeline = openvino_genai.LLMPipeline(
            str(exported_model_path(model_name)), device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate(self, input_text, max_new_tokens, system_prompt):
        prompt = chat_prompt(self.tokenizer, input_text, system_prompt)
        result = self.pipeline.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            apply_chat_template=False,
        )
        return str(result)
