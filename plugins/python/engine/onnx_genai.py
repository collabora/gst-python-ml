# ONNXGenAI
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

import ctypes
import functools
from pathlib import Path

from .hub_causal_lm import chat_prompt, export_once, hub_cache_name
from .ml_engine import MODEL_CACHE

GENAI_CACHE = MODEL_CACHE / "onnx-genai"
GENAI_CONFIG_NAME = "genai_config.json"
GENAI_LIBRARY_NAME = "libonnxruntime-genai.so"
DEVICE_BUILDS = {"cpu": ("cpu", "int4"), "cuda": ("cuda", "fp16")}


def device_build(device):
    build = next(
        (build for keyword, build in DEVICE_BUILDS.items() if keyword in device),
        None,
    )
    if build is None:
        raise ValueError(
            f"onnxruntime-genai builds for {', '.join(DEVICE_BUILDS)}, got {device}"
        )
    return build


def build_model(model_name, output, precision, execution_provider):
    from huggingface_hub.constants import HF_HUB_CACHE
    from onnxruntime_genai.models import builder

    arguments = (
        model_name,
        "",
        str(output),
        precision,
        execution_provider,
        HF_HUB_CACHE,
    )
    options = builder.parse_extra_options(*arguments, [])
    builder.create_model(*arguments, **options)


def built_model_path(model_name, device):
    execution_provider, precision = device_build(device)
    path = (
        GENAI_CACHE / f"{hub_cache_name(model_name)}-{execution_provider}-{precision}"
    )
    return export_once(
        path,
        GENAI_CONFIG_NAME,
        lambda output: build_model(model_name, output, precision, execution_provider),
    )


# genai's exit handler segfaults in the cuda provider unless OgaShutdown ran first
@functools.cache
def shut_down_genai_before_its_exit_handler():
    import onnxruntime_genai

    library_path = Path(onnxruntime_genai.__file__).parent / GENAI_LIBRARY_NAME
    if not library_path.exists():
        return
    shutdown = ctypes.cast(ctypes.CDLL(str(library_path)).OgaShutdown, ctypes.c_void_p)
    libc = ctypes.CDLL(None)
    libc.__cxa_atexit.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
    libc.__cxa_atexit(shutdown, None, None)


class GenAIModel:
    def __init__(self, model_name, device):
        import onnxruntime_genai as og
        from transformers import AutoTokenizer

        self.og = og
        self.model = og.Model(str(built_model_path(model_name, device)))
        shut_down_genai_before_its_exit_handler()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate(self, input_text, max_new_tokens, system_prompt):
        prompt = chat_prompt(self.tokenizer, input_text, system_prompt)
        input_ids = self.tokenizer(prompt)["input_ids"]
        params = self.og.GeneratorParams(self.model)
        params.set_search_options(
            max_length=len(input_ids) + max_new_tokens, do_sample=False
        )
        generator = self.og.Generator(self.model, params)
        generator.append_tokens(input_ids)
        tokens = []
        while not generator.is_done():
            generator.generate_next_token()
            tokens.append(generator.get_next_tokens()[0])
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
