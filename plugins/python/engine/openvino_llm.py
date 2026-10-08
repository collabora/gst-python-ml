# OpenVinoLLM
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
