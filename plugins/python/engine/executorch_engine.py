# ExecuTorchEngine
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


class ExecuTorchEngine(MLEngine):
    def __init__(self):
        super().__init__()
        self.model_type = None
        self.model_name = None
        self.kwargs = None
        self.backend = "cpu"

    def do_set_device(self, device):
        """Set ExecuTorch device/backend (cpu, xnnpack)."""
        self.device = device
        device_lower = (device or "cpu").lower()

        if device_lower in ("xnnpack", "qnn", "coreml", "mps"):
            self.backend = device_lower
            self.logger.info(f"ExecuTorch backend set to {device_lower}")
        elif device_lower in ("cuda", "gpu"):
            self.logger.warning(
                "ExecuTorch does not support CUDA directly. Falling back to CPU."
            )
            self.backend = "cpu"
            self.device = "cpu"
        else:
            self.backend = "cpu"
            self.logger.info("ExecuTorch backend set to CPU")

    def do_load_model(self, model_name, **kwargs):
        """Load an ExecuTorch .pte model file."""
        self.model_name = model_name
        self.kwargs = kwargs

        if not os.path.isfile(model_name) or not model_name.endswith(".pte"):
            self.logger.error(
                f"ExecuTorch requires a .pte model file, got: {model_name}"
            )
            return False

        try:
            from executorch.runtime import Runtime

            runtime = Runtime.get()
            program = runtime.load_program(model_name)
            self.model = program.load_method("forward")
            self.model_type = "pte"
            self.logger.info(f"ExecuTorch model loaded from: {model_name}")
            return True
        except ImportError:
            self.logger.error(
                "executorch is not installed. " "Install with: pip install executorch"
            )
            return False
        except Exception as e:
            self.logger.error(f"Error loading ExecuTorch model '{model_name}': {e}")
            self.model = None
            return False

    def do_forward(self, frames):
        """Execute inference through the ExecuTorch module."""
        if self.model is None:
            self.logger.error("No model loaded.")
            return None

        is_batch = isinstance(frames, np.ndarray) and frames.ndim == 4
        if not isinstance(frames, np.ndarray):
            self.logger.error(f"Invalid input type for forward: {type(frames)}")
            return None

        try:
            import torch
        except ImportError:
            self.logger.error("torch is required for ExecuTorch tensor conversion.")
            return None

        img = self._apply_input_format(frames.astype(np.float32) / 255.0, is_batch)
        # the runtime reads the buffer as laid out, a transposed view crashes it
        input_tensor = torch.from_numpy(np.ascontiguousarray(img))

        try:
            outputs = self.model.execute([input_tensor])
            if isinstance(outputs, (list, tuple)):
                raw = (
                    outputs[0].numpy()
                    if hasattr(outputs[0], "numpy")
                    else np.array(outputs[0])
                )
            else:
                raw = (
                    outputs.numpy() if hasattr(outputs, "numpy") else np.array(outputs)
                )

            return self._apply_post_process(raw, is_batch)
        except Exception as e:
            self.logger.error(f"ExecuTorch inference failed: {e}")
            return None

    def do_generate(self, input_text, max_length=1000, system_prompt=None):
        """Text generation is not supported by ExecuTorch engine."""
        raise NotImplementedError(
            "ExecuTorch engine does not support text generation. "
            "Export an LLM to .pte and use do_forward for token-level inference."
        )
