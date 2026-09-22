# CandleEngine
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


class CandleEngine(MLEngine):
    def __init__(self):
        super().__init__()
        self.model_type = None
        self.model_name = None
        self.kwargs = None
        self.candle_device = None

    def do_set_device(self, device):
        """Set Candle device (cpu or cuda)."""
        try:
            import candle
        except ImportError:
            self.logger.error(
                "candle is not installed. "
                "Install the candle Python bindings (pip install candle-nn)."
            )
            return

        self.device = device
        device_lower = (device or "cpu").lower()

        if device_lower in ("cuda", "gpu"):
            try:
                self.candle_device = candle.Device.cuda(0)
                self.logger.info("Candle device set to CUDA:0")
            except Exception:
                self.logger.warning("CUDA not available in Candle, falling back to CPU")
                self.candle_device = candle.Device.cpu()
                self.device = "cpu"
        else:
            self.candle_device = candle.Device.cpu()
            self.logger.info("Candle device set to CPU")

    def do_load_model(self, model_name, **kwargs):
        """Load a safetensors model via Candle."""
        self.model_name = model_name
        self.kwargs = kwargs

        try:
            import candle
        except ImportError:
            self.logger.error("candle is not installed.")
            return False

        # Local safetensors file
        if os.path.isfile(model_name) and model_name.endswith(".safetensors"):
            self.model = candle.load_safetensors(model_name)
            self.model_type = "custom"
            self.logger.info(f"Candle model loaded from safetensors: {model_name}")
            return True

        # Directory containing safetensors + config
        if os.path.isdir(model_name):
            st_files = [f for f in os.listdir(model_name) if f.endswith(".safetensors")]
            if not st_files:
                self.logger.error(
                    f"No .safetensors files found in directory: {model_name}"
                )
                return False
            weights = {}
            for st_file in st_files:
                path = os.path.join(model_name, st_file)
                loaded = candle.load_safetensors(path)
                weights.update(loaded)
            self.model = weights
            self.model_type = "custom"
            self.logger.info(
                f"Candle model loaded from directory: {model_name} "
                f"({len(st_files)} safetensors files)"
            )
            return True

        # Try HuggingFace Hub download
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(repo_id=model_name, filename="model.safetensors")
            self.model = candle.load_safetensors(path)
            self.model_type = "custom"
            self.logger.info(f"Candle model downloaded from HuggingFace: {model_name}")
            return True
        except ImportError:
            self.logger.error(
                "huggingface_hub is not installed for remote model download."
            )
            return False

    def do_forward(self, frames):
        """Run inference through Candle model weights."""
        if self.model is None:
            self.logger.error("No model loaded.")
            return None

        is_batch = isinstance(frames, np.ndarray) and frames.ndim == 4
        if not isinstance(frames, np.ndarray):
            self.logger.error(f"Invalid input type for forward: {type(frames)}")
            return None

        try:
            import candle
        except ImportError:
            self.logger.error("candle is not installed.")
            return None

        img = self._apply_input_format(frames.astype(np.float32) / 255.0, is_batch)

        input_tensor = candle.Tensor(img)
        if self.candle_device:
            input_tensor = input_tensor.to_device(self.candle_device)

        # For weight-dict models, return the tensor for downstream processing
        if isinstance(self.model, dict):
            raw = np.array(input_tensor.to_dtype(candle.f32).values())
        elif callable(self.model):
            output = self.model(input_tensor)
            raw = np.array(output.values())
        else:
            raw = np.array(input_tensor.values())

        return self._apply_post_process(raw, is_batch)

    def do_generate(self, input_text, max_length=1000, system_prompt=None):
        """Basic text generation via Candle model (if supported)."""
        if self.model is None:
            self.logger.error("No model loaded.")
            return None

        self.logger.warning(
            "Candle text generation requires a model-specific generation loop. "
            "This engine provides basic weight loading only. "
            "Consider using a dedicated Candle model wrapper."
        )
        raise NotImplementedError(
            "Generic text generation is not supported by CandleEngine. "
            "Implement a model-specific generation loop using the loaded weights."
        )
