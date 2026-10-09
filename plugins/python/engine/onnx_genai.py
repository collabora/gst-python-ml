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
import io
import wave
from pathlib import Path

from .hub_causal_lm import chat_prompt, export_once, hub_cache_name
from .ml_engine import MODEL_CACHE
from .whisper_engine import WHISPER_SAMPLE_RATE

GENAI_CACHE = MODEL_CACHE / "onnx-genai"
GENAI_CONFIG_NAME = "genai_config.json"
GENAI_LIBRARY_NAME = "libonnxruntime-genai.so"
DEVICE_BUILDS = {"cpu": ("cpu", "int4"), "cuda": ("cuda", "fp16")}
# int4 makes whisper-tiny repeat itself
WHISPER_DEVICE_BUILDS = {"cpu": ("cpu", "fp32"), "cuda": ("cuda", "fp16")}
WHISPER_CONTEXT_TOKENS = 448
WHISPER_START_TOKEN = "<|startoftranscript|>"
WHISPER_PREVIOUS_TOKEN = "<|startofprev|>"
WHISPER_NO_TIMESTAMPS_TOKEN = "<|notimestamps|>"
PCM16_SAMPLE_WIDTH = 2
PCM16_SCALE = 32767


def device_build(device, builds=DEVICE_BUILDS):
    build = next(
        (build for keyword, build in builds.items() if keyword in device),
        None,
    )
    if build is None:
        raise ValueError(
            f"onnxruntime-genai builds for {', '.join(builds)}, got {device}"
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


def built_model_path(model_name, device, builds=DEVICE_BUILDS):
    execution_provider, precision = device_build(device, builds)
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


def whisper_prompt(language, task, initial_prompt):
    prompt = f"{WHISPER_PREVIOUS_TOKEN} {initial_prompt}" if initial_prompt else ""
    prompt += WHISPER_START_TOKEN
    if language:
        prompt += f"<|{language}|>"
    return f"{prompt}<|{task}|>{WHISPER_NO_TIMESTAMPS_TOKEN}"


# genai decodes audio from an encoded file, not from samples
def wav_bytes(audio):
    import numpy as np

    samples = (np.clip(audio, -1.0, 1.0) * PCM16_SCALE).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(PCM16_SAMPLE_WIDTH)
        writer.setframerate(WHISPER_SAMPLE_RATE)
        writer.writeframes(samples.tobytes())
    return buffer.getvalue()


class GenAIWhisper:
    def __init__(self, model_name, device):
        import onnxruntime_genai as og

        self.og = og
        self.model = og.Model(
            str(built_model_path(model_name, device, WHISPER_DEVICE_BUILDS))
        )
        shut_down_genai_before_its_exit_handler()
        self.processor = self.model.create_multimodal_processor()
        self.tokenizer = og.Tokenizer(self.model)

    def transcribe(self, audio, language, task, beam_size, initial_prompt):
        prompt = whisper_prompt(language, task, initial_prompt)
        prompt_length = len(self.tokenizer.encode(prompt))
        inputs = self.processor(
            [prompt], audios=self.og.Audios.open_bytes(wav_bytes(audio))
        )
        params = self.og.GeneratorParams(self.model)
        params.set_search_options(
            do_sample=False,
            num_beams=beam_size,
            max_length=WHISPER_CONTEXT_TOKENS,
            batch_size=1,
        )
        generator = self.og.Generator(self.model, params)
        generator.set_inputs(inputs)
        while not generator.is_done():
            generator.generate_next_token()
        tokens = generator.get_sequence(0)[prompt_length:]
        return [self.processor.decode(tokens).strip()]
