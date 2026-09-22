# MiGraphXEngine
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
import migraphx

from .ml_engine import MLEngine


class MiGraphXEngine(MLEngine):
    def __init__(self):
        super().__init__()
        self.program = None
        self.target = None
        self.input_names = None
        self.output_shapes = None
        self.model_name = None
        self.kwargs = None
        self.fp16 = False

    def _input_is_nchw(self):
        """Auto-detect whether the model's first input expects NCHW layout."""
        if self.program is None:
            return False
        param_shapes = self.program.get_parameter_shapes()
        if not param_shapes:
            return False
        first_shape = next(iter(param_shapes.values()))
        lens = first_shape.lens()
        return len(lens) == 4 and lens[1] in (1, 3, 4)

    def do_load_model(self, model_name, **kwargs):
        """Load an ONNX model via MiGraphX and compile it for the target device."""
        self.model_name = model_name
        self.kwargs = kwargs
        self.fp16 = kwargs.get("fp16", False)

        if not os.path.isfile(model_name):
            self.logger.error(
                f"MiGraphX requires an ONNX model file path, got: {model_name}"
            )
            return False

        if not model_name.endswith(".onnx"):
            self.logger.warning(f"MiGraphX expects an .onnx file, got: {model_name}")

        # Parse ONNX model
        parse_kwargs = {}
        if "default_dim_value" in kwargs:
            parse_kwargs["default_dim_value"] = kwargs["default_dim_value"]
        if "map_input_dims" in kwargs:
            parse_kwargs["map_input_dims"] = kwargs["map_input_dims"]

        self.program = migraphx.parse_onnx(model_name, **parse_kwargs)

        # Optional fp16 quantization
        if self.fp16:
            migraphx.quantize_fp16(self.program)

        # Compile for target
        target = self.target or migraphx.get_target("gpu")
        offload_copy = kwargs.get("offload_copy", True)
        self.program.compile(target, offload_copy=offload_copy)

        # Cache parameter info
        self.input_names = self.program.get_parameter_names()
        self.output_shapes = self.program.get_output_shapes()
        self.model = self.program

        self.logger.info(
            f"MiGraphX model loaded and compiled: {model_name} "
            f"(inputs: {self.input_names}, fp16: {self.fp16})"
        )
        return True

    def do_set_device(self, device):
        """Set the MiGraphX compilation target."""
        self.device = device

        if device == "cpu":
            self.target = migraphx.get_target("ref")
            self.logger.info("MiGraphX target set to CPU (ref)")
        elif "rocm" in device or "gpu" in device or "hip" in device:
            self.target = migraphx.get_target("gpu")
            self.logger.info("MiGraphX target set to GPU")
        else:
            self.logger.warning(
                f"Unknown device '{device}' for MiGraphX, defaulting to GPU"
            )
            self.target = migraphx.get_target("gpu")

        # Reload model if already loaded
        if self.model_name:
            self.do_load_model(self.model_name, **(self.kwargs or {}))

    def do_forward(self, frames):
        """Run inference on a single frame or batch of frames."""
        if self.program is None:
            self.logger.error("No model loaded")
            return None

        is_batch = isinstance(frames, np.ndarray) and frames.ndim == 4

        # Apply input format (NCHW/NHWC)
        fmt = self.input_format
        if fmt == "auto" and self._input_is_nchw():
            self.input_format = "nchw"
        img = self._apply_input_format(frames.astype(np.float32) / 255.0, is_batch)

        # Build parameter dict — map first input name to the data
        params = {self.input_names[0]: migraphx.argument(np.ascontiguousarray(img))}

        # Run inference
        results = self.program.run(params)

        # Convert results to numpy
        outputs = [np.array(r) for r in results]
        raw = outputs if len(outputs) > 1 else outputs[0]

        return self._apply_post_process(raw, is_batch)

    def do_generate(self, input_text, max_length=1000, system_prompt=None):
        """MiGraphX does not support text generation."""
        raise NotImplementedError(
            "MiGraphX is an inference-only engine and does not support text generation. "
            "Use PyTorch or llama.cpp for LLM workloads."
        )
