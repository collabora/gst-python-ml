# MLXEngine
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


class MLXEngine(MLEngine):
    def __init__(self):
        super().__init__()
        self.model_type = None
        self.model_name = None
        self.kwargs = None

    def do_set_device(self, device):
        """Set MLX device (gpu or cpu)."""
        try:
            import mlx.core as mx
        except ImportError:
            self.logger.error("mlx is not installed. Install with: pip install mlx")
            return

        self.device = device
        device_lower = (device or "gpu").lower()

        if device_lower in ("gpu", "cuda", "metal"):
            mx.set_default_device(mx.gpu)
            self.logger.info("MLX device set to GPU (Metal)")
        else:
            mx.set_default_device(mx.cpu)
            self.logger.info("MLX device set to CPU")

    def do_load_model(self, model_name, **kwargs):
        """Load a model via MLX from local files, HuggingFace via mlx-lm, or PyTorch conversion."""
        self.model_name = model_name
        self.kwargs = kwargs

        try:
            # Local SafeTensors / npz model
            if os.path.isfile(model_name) and model_name.endswith(
                (".safetensors", ".npz")
            ):
                import mlx.core as mx

                if model_name.endswith(".npz"):
                    self.model = dict(np.load(model_name))
                    self.model = {k: mx.array(v) for k, v in self.model.items()}
                else:
                    from mlx.utils import load

                    self.model = load(model_name)
                self.model_type = "custom"
                self.logger.info(f"MLX model loaded from local path: {model_name}")
                return True

            from torchvision import models as tv_models

            if hasattr(tv_models, model_name):
                pt_model = getattr(tv_models, model_name)(pretrained=True)
                if not isinstance(pt_model, tv_models.ResNet):
                    self.logger.error(
                        f"MLX runs the torchvision resnet family, not '{model_name}'."
                    )
                    return False
                from .mlx_resnet import mlx_resnet

                self.model = mlx_resnet(pt_model.eval())
                self.model_type = "classification"
                self.logger.info(
                    f"Pre-trained vision model '{model_name}' loaded with MLX."
                )
                return True

            # LLM via mlx-lm
            try:
                from mlx_lm import load as mlx_lm_load

                self.model, self.tokenizer = mlx_lm_load(model_name)
                self.model_type = "llm"
                self.logger.info(f"LLM model '{model_name}' loaded via mlx-lm.")
                return True
            except ImportError:
                self.logger.info("mlx-lm not available, trying PyTorch conversion.")
            except Exception as e:
                self.logger.info(
                    f"mlx-lm load failed ({e}), trying PyTorch conversion."
                )

            # Convert from PyTorch/HuggingFace
            import mlx.core as mx
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            pt_model = AutoModelForCausalLM.from_pretrained(model_name)
            pt_model.eval()
            state_dict = pt_model.state_dict()
            self.model = {k: mx.array(v.cpu().numpy()) for k, v in state_dict.items()}
            self.model_type = "llm_converted"
            self.logger.info(
                f"Model '{model_name}' converted from PyTorch to MLX arrays."
            )
            return True

        except Exception as e:
            self.logger.error(f"Error loading model '{model_name}': {e}")
            self.model = None
            self.tokenizer = None
            return False

    def do_forward(self, frames):
        """Execute inference by converting numpy input to MLX arrays."""
        try:
            import mlx.core as mx
        except ImportError:
            self.logger.error("mlx is not installed.")
            return None

        is_batch = isinstance(frames, np.ndarray) and frames.ndim == 4
        if not isinstance(frames, np.ndarray):
            self.logger.error(f"Invalid input type for forward: {type(frames)}")
            return None

        if self.model_type == "llm" or self.model_type == "llm_converted":
            self.logger.warning(
                "do_forward is not applicable for LLM models. Use do_generate instead."
            )
            return None

        img = self._apply_input_format(frames.astype(np.float32) / 255.0, is_batch)
        mx_input = mx.array(img)
        if self.model_type == "classification":
            preds = np.array(self.model(mx_input))
            probs = np.exp(preds) / np.sum(np.exp(preds), axis=1, keepdims=True)
            top_classes = np.argmax(probs, axis=1)
            confidences = np.max(probs, axis=1)
            results = [
                {"labels": [int(c)], "scores": [float(s)]}
                for c, s in zip(top_classes, confidences)
            ]
            return results[0] if not is_batch else results
        if self.model_type == "custom" and callable(self.model):
            return self._apply_post_process(np.array(self.model(mx_input)), is_batch)
        self.logger.error("A bare state dict has no graph to run, load a model.")
        return None

    def do_generate(self, input_text, max_length=1000, system_prompt=None):
        """Generate text using mlx-lm for LLM models."""
        if self.model_type == "llm":
            try:
                from mlx_lm import generate

                prompt = input_text
                if system_prompt:
                    prompt = f"{system_prompt}\n\n{input_text}"

                result = generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=max_length,
                )
                self.logger.info(f"Generated text: {result[:100]}...")
                return result
            except ImportError:
                self.logger.error("mlx-lm is not installed for generation.")
                return None
            except Exception as e:
                self.logger.error(f"MLX generation failed: {e}")
                return None

        elif self.model_type == "llm_converted":
            self.logger.error(
                "Text generation for converted PyTorch models requires mlx-lm. "
                "Load the model directly via mlx-lm instead."
            )
            return None

        raise ValueError("Generate is only supported for LLM models.")
