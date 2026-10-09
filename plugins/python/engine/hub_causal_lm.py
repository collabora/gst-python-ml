# HubCausalLM
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

import os
import tempfile
from pathlib import Path

CAUSAL_LM_SUFFIX = "ForCausalLM"
WHISPER_ARCHITECTURE = "WhisperForConditionalGeneration"
HUB_NAME_SEPARATOR = "/"
CACHE_NAME_SEPARATOR = "--"


def hub_architectures(model_name):
    if HUB_NAME_SEPARATOR not in model_name:
        return []
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(model_name).architectures or []


def is_hub_causal_lm(model_name):
    return any(
        name.endswith(CAUSAL_LM_SUFFIX) for name in hub_architectures(model_name)
    )


def is_hub_whisper(model_name):
    return WHISPER_ARCHITECTURE in hub_architectures(model_name)


def hub_cache_name(model_name):
    return model_name.replace(HUB_NAME_SEPARATOR, CACHE_NAME_SEPARATOR)


# a killed export leaves no half-written model behind
def export_once(path, exported_file_name, export):
    if (path / exported_file_name).exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as work_directory:
        output = Path(work_directory) / path.name
        export(output)
        os.replace(output, path)
    return path


# base models such as phi-2 ship no chat template
def chat_prompt(tokenizer, input_text, system_prompt):
    if tokenizer.chat_template is None:
        return f"{system_prompt}\n{input_text}" if system_prompt else input_text
    messages = [{"role": "user", "content": input_text}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
