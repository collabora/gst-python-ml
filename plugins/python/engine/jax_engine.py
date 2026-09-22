# JAXEngine
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


class JAXEngine(MLEngine):
    def __init__(self):
        super().__init__()
        self.model_type = None
        self.model_name = None
        self.kwargs = None
        self.params = None
        self.apply_fn = None

    def do_set_device(self, device):
        """Set JAX default device (cpu, gpu, tpu)."""
        try:
            import jax
        except ImportError:
            self.logger.error("jax is not installed. Install with: pip install jax")
            return

        self.device = device
        device_lower = (device or "cpu").lower()

        if device_lower in ("gpu", "cuda"):
            devices = jax.devices("gpu")
            if devices:
                jax.default_device = devices[0]
                self.logger.info(f"JAX device set to GPU: {devices[0]}")
            else:
                self.logger.warning("No GPU available, falling back to CPU")
                self.device = "cpu"
                jax.default_device = jax.devices("cpu")[0]
        elif device_lower == "tpu":
            devices = jax.devices("tpu")
            if devices:
                jax.default_device = devices[0]
                self.logger.info(f"JAX device set to TPU: {devices[0]}")
            else:
                self.logger.warning("No TPU available, falling back to CPU")
                self.device = "cpu"
                jax.default_device = jax.devices("cpu")[0]
        else:
            jax.default_device = jax.devices("cpu")[0]
            self.logger.info("JAX device set to CPU")

    def do_load_model(self, model_name, **kwargs):
        """Load a Flax model from HuggingFace or local checkpoint."""
        self.model_name = model_name
        self.kwargs = kwargs
        tokenizer_name = kwargs.get("tokenizer_name")

        # Local Flax checkpoint directory
        if os.path.isdir(model_name):
            return self._load_local_checkpoint(model_name)

        # Local .msgpack file
        if os.path.isfile(model_name) and model_name.endswith(".msgpack"):
            return self._load_msgpack(model_name)

        from torchvision import models as tv_models

        if hasattr(tv_models, model_name):
            pt_model = getattr(tv_models, model_name)(pretrained=True)
            if not isinstance(pt_model, tv_models.ResNet):
                self.logger.error(
                    f"JAX runs the torchvision resnet family, not '{model_name}'."
                )
                return False
            from .jax_resnet import jax_resnet

            self.model = jax_resnet(pt_model.eval())
            self.model_type = "classification"
            self.logger.info(
                f"Pre-trained vision model '{model_name}' compiled with JAX."
            )
            return True

        # HuggingFace Flax model
        return self._load_from_huggingface(model_name, tokenizer_name)

    def _load_local_checkpoint(self, model_dir):
        """Load Flax params from a local checkpoint directory."""
        try:
            from flax.serialization import from_bytes
        except ImportError:
            self.logger.error("flax is not installed. Install with: pip install flax")
            return False

        msgpack_path = os.path.join(model_dir, "flax_model.msgpack")
        if not os.path.isfile(msgpack_path):
            self.logger.error(f"No flax_model.msgpack found in {model_dir}")
            return False

        with open(msgpack_path, "rb") as f:
            self.params = from_bytes(None, f.read())
        self.model_type = "custom"
        self.logger.info(f"Flax checkpoint loaded from: {model_dir}")
        return True

    def _load_msgpack(self, path):
        """Load Flax params from a .msgpack file."""
        try:
            from flax.serialization import from_bytes
        except ImportError:
            self.logger.error("flax is not installed.")
            return False

        with open(path, "rb") as f:
            self.params = from_bytes(None, f.read())
        self.model_type = "custom"
        self.logger.info(f"Flax params loaded from: {path}")
        return True

    def _load_from_huggingface(self, model_name, tokenizer_name):
        """Load a Flax model from HuggingFace Transformers."""
        try:
            from transformers import (
                AutoTokenizer,
                FlaxAutoModelForCausalLM,
                FlaxAutoModel,
            )
        except ImportError:
            self.logger.error(
                "transformers with Flax support is not installed. "
                "Install with: pip install transformers[flax]"
            )
            return False

        tok_name = tokenizer_name or model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tok_name)

        # Try causal LM first, then generic model
        try:
            self.model = FlaxAutoModelForCausalLM.from_pretrained(model_name)
            self.params = self.model.params
            self.apply_fn = self.model
            self.model_type = "llm"
            self.logger.info(f"Flax causal LM '{model_name}' loaded from HuggingFace.")
            return True
        except Exception:
            pass

        self.model = FlaxAutoModel.from_pretrained(model_name)
        self.params = self.model.params
        self.apply_fn = self.model
        self.model_type = "encoder"
        self.logger.info(f"Flax encoder model '{model_name}' loaded from HuggingFace.")
        return True

    def do_forward(self, frames):
        """Execute inference by converting numpy to JAX arrays."""
        if self.params is None and self.model is None:
            self.logger.error("No model loaded.")
            return None

        is_batch = isinstance(frames, np.ndarray) and frames.ndim == 4
        if not isinstance(frames, np.ndarray):
            self.logger.error(f"Invalid input type for forward: {type(frames)}")
            return None

        try:
            import jax.numpy as jnp
        except ImportError:
            self.logger.error("jax is not installed.")
            return None

        img = self._apply_input_format(frames.astype(np.float32) / 255.0, is_batch)
        jax_input = jnp.array(img)
        if self.model_type == "classification":
            preds = np.asarray(self.model(jax_input))
            probs = np.exp(preds) / np.sum(np.exp(preds), axis=1, keepdims=True)
            top_classes = np.argmax(probs, axis=1)
            confidences = np.max(probs, axis=1)
            results = [
                {"labels": [int(c)], "scores": [float(s)]}
                for c, s in zip(top_classes, confidences)
            ]
            return results[0] if not is_batch else results
        if self.apply_fn is None:
            self.logger.error(
                "A bare Flax checkpoint has no graph to run, load a model."
            )
            return None
        outputs = self.apply_fn(pixel_values=jax_input, params=self.params)
        raw = np.array(
            outputs.logits if hasattr(outputs, "logits") else outputs.last_hidden_state
        )
        return self._apply_post_process(raw, is_batch)

    def do_generate(self, input_text, max_length=1000, system_prompt=None):
        """Generate text using Flax model's generate method."""
        if self.model_type != "llm":
            raise ValueError("Generate is only supported for LLM models.")

        if self.model is None or self.tokenizer is None:
            self.logger.error("No LLM model or tokenizer loaded.")
            return None

        prompt = input_text
        if system_prompt:
            prompt = f"{system_prompt}\n\n{input_text}"

        inputs = self.tokenizer(prompt, return_tensors="jax")
        output_ids = self.model.generate(
            **inputs,
            max_length=max_length,
            params=self.params,
        )
        generated_text = self.tokenizer.decode(
            output_ids.sequences[0], skip_special_tokens=True
        )
        self.logger.info(f"Generated text: {generated_text[:100]}...")
        return generated_text
