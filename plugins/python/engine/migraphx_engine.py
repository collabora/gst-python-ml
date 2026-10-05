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

import hashlib
import os
from pathlib import Path

import numpy as np
import migraphx

from .ml_engine import MLEngine, fixed_height_width

COMPILED_PROGRAM_CACHE = Path.home() / ".cache" / "gst-python-ml" / "migraphx"
PARTIAL_SAVE_SUFFIX = ".partial"


class MiGraphXEngine(MLEngine):
    def __init__(self):
        super().__init__()
        self.program = None
        self.target_name = "gpu"
        self.input_names = None
        self.output_shapes = None
        self.model_name = None
        self.kwargs = None
        self.fp16 = False
        self.dynamic_input_name = None
        self.dynamic_input_dims = None
        self.compiled_input_shape = None

    def _input_is_nchw(self):
        """Auto-detect whether the model's first input expects NCHW layout."""
        if self.dynamic_input_dims is not None:
            lens = self.dynamic_input_dims
        elif self.program is None:
            return False
        else:
            param_shapes = self.program.get_parameter_shapes()
            if not param_shapes:
                return False
            lens = next(iter(param_shapes.values())).lens()
        return len(lens) == 4 and lens[1] in (1, 3, 4)

    def _model_input_hw(self):
        if self.program is None or self.dynamic_input_name is not None:
            return None
        shapes = self.program.get_parameter_shapes()
        return fixed_height_width(shapes[self.input_names[0]].lens())

    def do_load_model(self, model_name, **kwargs):
        """Load an ONNX model via MiGraphX and compile it for the target device."""
        self.model_name = model_name
        self.kwargs = kwargs
        self.fp16 = kwargs.get("fp16", False)

        if not os.path.isfile(model_name):
            raise FileNotFoundError(
                f"MiGraphX requires an ONNX model file path, got: {model_name}"
            )

        if not model_name.endswith(".onnx"):
            self.logger.warning(f"MiGraphX expects an .onnx file, got: {model_name}")

        self.program = None
        self.compiled_input_shape = None
        self.dynamic_input_name = None
        self.dynamic_input_dims = None
        input_name, input_dims = self._first_input(model_name)
        height_or_width_is_dynamic = len(input_dims) == 4 and None in input_dims[2:]
        # migraphx parses a dynamic height or width at a default size that can divide by zero
        if height_or_width_is_dynamic and "map_input_dims" not in kwargs:
            self.dynamic_input_name = input_name
            self.dynamic_input_dims = input_dims
            self.logger.info(
                f"MiGraphX model {model_name} has a dynamic input size, "
                "compiling on the first frame"
            )
            return True

        self._parse_and_compile(self._parse_kwargs())
        return True

    def _first_input(self, model_name):
        import onnx

        graph = onnx.load(model_name, load_external_data=False).graph
        initializer_names = {initializer.name for initializer in graph.initializer}
        first_input = next(
            graph_input
            for graph_input in graph.input
            if graph_input.name not in initializer_names
        )
        dims = [
            None if dim.dim_param or dim.dim_value == 0 else dim.dim_value
            for dim in first_input.type.tensor_type.shape.dim
        ]
        return first_input.name, dims

    def _parse_kwargs(self):
        return {
            key: self.kwargs[key]
            for key in ("default_dim_value", "map_input_dims")
            if key in self.kwargs
        }

    def _compiled_program_path(self, parse_kwargs, offload_copy):
        model_path = Path(self.model_name).resolve()
        model_stat = model_path.stat()
        key = repr(
            (
                str(model_path),
                model_stat.st_size,
                model_stat.st_mtime_ns,
                self.target_name,
                self.fp16,
                offload_copy,
                sorted(parse_kwargs.items()),
            )
        )
        digest = hashlib.sha256(key.encode()).hexdigest()
        return COMPILED_PROGRAM_CACHE / f"{digest}.mxr"

    def _parse_and_compile(self, parse_kwargs):
        offload_copy = self.kwargs.get("offload_copy", True)
        program_path = self._compiled_program_path(parse_kwargs, offload_copy)
        loaded_from_cache = program_path.is_file()
        if loaded_from_cache:
            self.program = migraphx.load(str(program_path))
        else:
            self.program = migraphx.parse_onnx(self.model_name, **parse_kwargs)
            if self.fp16:
                migraphx.quantize_fp16(self.program)
            self.program.compile(
                migraphx.get_target(self.target_name), offload_copy=offload_copy
            )
            COMPILED_PROGRAM_CACHE.mkdir(parents=True, exist_ok=True)
            partial_path = program_path.with_suffix(PARTIAL_SAVE_SUFFIX)
            migraphx.save(self.program, str(partial_path))
            partial_path.rename(program_path)

        # Cache parameter info
        self.input_names = self.program.get_parameter_names()
        self.output_shapes = self.program.get_output_shapes()
        self.model = self.program

        program_source = (
            f"loaded from cache {program_path}" if loaded_from_cache else "compiled"
        )
        self.logger.info(
            f"MiGraphX model {program_source}: {self.model_name} "
            f"(inputs: {self.input_names}, fp16: {self.fp16})"
        )

    def do_set_device(self, device):
        """Set the MiGraphX compilation target."""
        if device == "cpu":
            self.target_name = "ref"
            self.logger.info("MiGraphX target set to CPU (ref)")
        elif "rocm" in device or "gpu" in device or "hip" in device:
            self.target_name = "gpu"
            self.logger.info("MiGraphX target set to GPU")
        else:
            raise ValueError(f"Invalid device specified: {device}")
        self.device = device

        # Reload model if already loaded
        if self.model_name:
            self.do_load_model(self.model_name, **(self.kwargs or {}))

    def do_forward(self, frames):
        """Run inference on a single frame or batch of frames."""
        if self.program is None and self.dynamic_input_name is None:
            self.logger.error("No model loaded")
            return None

        is_batch = isinstance(frames, np.ndarray) and frames.ndim == 4

        # Apply input format (NCHW/NHWC)
        fmt = self.input_format
        if fmt == "auto" and self._input_is_nchw():
            self.input_format = "nchw"
        resized, transform = self._letterbox(frames, is_batch)
        img = self._apply_input_format(resized.astype(np.float32) / 255.0, is_batch)

        # Build parameter dict — map first input name to the data
        # migraphx reads the array without keeping it alive
        model_input = np.ascontiguousarray(img)
        if (
            self.dynamic_input_name is not None
            and model_input.shape != self.compiled_input_shape
        ):
            parse_kwargs = self._parse_kwargs()
            parse_kwargs["map_input_dims"] = {
                self.dynamic_input_name: list(model_input.shape)
            }
            self._parse_and_compile(parse_kwargs)
            self.compiled_input_shape = model_input.shape
        params = {self.input_names[0]: migraphx.argument(model_input)}

        # Run inference
        results = self.program.run(params)

        # Convert results to numpy
        outputs = [np.array(r) for r in results]
        raw = outputs if len(outputs) > 1 else outputs[0]

        results = self._apply_post_process(raw, is_batch)
        if transform is not None:
            self._unletterbox(results, transform)
        return results

    def do_generate(self, input_text, max_length=1000, system_prompt=None):
        """MiGraphX does not support text generation."""
        raise NotImplementedError(
            "MiGraphX is an inference-only engine and does not support text generation. "
            "Use PyTorch or llama.cpp for LLM workloads."
        )
