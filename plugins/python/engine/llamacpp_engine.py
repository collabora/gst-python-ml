# LlamaCppEngine
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
import numpy as np

from .ml_engine import MLEngine

GGUF_SUFFIX = ".gguf"
# model-name=repo:quant, the shorthand llama.cpp's own -hf flag takes
HUB_QUANT_SEPARATOR = ":"


def hub_gguf_pattern(model_name):
    repo_id, _, quant = model_name.partition(HUB_QUANT_SEPARATOR)
    return repo_id, f"*{quant}{GGUF_SUFFIX}"


class LlamaCppEngine(MLEngine):
    def __init__(self):
        super().__init__()
        self.model_type = None
        self.model_name = None
        self.kwargs = None
        self.n_gpu_layers = 0
        self.n_ctx = 4096

    def do_set_device(self, device):
        """Set n_gpu_layers based on device string."""
        self.device = device
        device_lower = (device or "cpu").lower()

        if device_lower in ("cuda", "gpu", "metal"):
            self.n_gpu_layers = -1  # Offload all layers to GPU
            self.logger.info(
                f"llama.cpp will offload all layers to GPU ({device_lower})"
            )
        elif device_lower == "cpu":
            self.n_gpu_layers = 0
            self.logger.info("llama.cpp set to CPU-only inference")
        else:
            # Treat as integer number of GPU layers
            if not device_lower.isdigit():
                raise ValueError(
                    f"Invalid device specified: {device}, "
                    "use cpu, cuda, metal or a number of GPU layers"
                )
            self.n_gpu_layers = int(device_lower)
            self.logger.info(
                f"llama.cpp will offload {self.n_gpu_layers} layers to GPU"
            )

        # Reload model with new GPU layer config if already loaded
        if self.model is not None and self.model_name:
            self.do_load_model(self.model_name, **(self.kwargs or {}))

    def do_load_model(self, model_name, **kwargs):
        """Load a GGUF model file via llama-cpp-python."""
        self.model_name = model_name
        self.kwargs = kwargs
        self.n_ctx = kwargs.get("n_ctx", self.n_ctx)

        from llama_cpp import Llama

        options = {
            "n_gpu_layers": self.n_gpu_layers,
            "n_ctx": self.n_ctx,
            "verbose": False,
        }
        if os.path.isfile(model_name):
            self.model = Llama(model_path=model_name, **options)
        elif model_name.endswith(GGUF_SUFFIX):
            raise FileNotFoundError(f"GGUF model file not found: {model_name}")
        else:
            repo_id, pattern = hub_gguf_pattern(model_name)
            self.model = Llama.from_pretrained(repo_id, pattern, **options)
        self.model_type = "llm"
        self.logger.info(
            f"GGUF model loaded: {model_name} "
            f"(n_gpu_layers={self.n_gpu_layers}, n_ctx={self.n_ctx})"
        )
        return True

    def do_forward(self, frames):
        """Forward pass is not applicable for llama.cpp LLMs."""
        if not isinstance(frames, np.ndarray):
            self.logger.error(f"Invalid input type for forward: {type(frames)}")
            return None

        self.logger.warning(
            "do_forward is not applicable for llama.cpp LLM models. "
            "Use do_generate for text generation."
        )
        return None

    def do_generate(self, input_text, max_length=1000, system_prompt=None):
        """Generate text using the llama.cpp model."""
        if self.model is None:
            self.logger.error("No model loaded.")
            return None

        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text},
            ]
            response = self.model.create_chat_completion(
                messages=messages,
                max_tokens=max_length,
            )
            result = response["choices"][0]["message"]["content"]
        else:
            response = self.model(
                input_text,
                max_tokens=max_length,
                echo=False,
            )
            result = response["choices"][0]["text"]

        self.logger.info(f"Generated text: {result[:100]}...")
        return result
