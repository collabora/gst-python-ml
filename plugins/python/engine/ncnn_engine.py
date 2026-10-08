# NCNNEngine
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
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import ncnn

from .ml_engine import MLEngine, converted_model_path

ENGINE_NAME = "ncnn"
ONNX_SUFFIX = ".onnx"
PARAM_SUFFIX = ".param"
BIN_SUFFIX = ".bin"
PNNX_SOURCE_NAME = "model.onnx"
PNNX_PARAM_NAME = "model.ncnn.param"
PNNX_BIN_NAME = "model.ncnn.bin"
PNNX_INPUT_TYPE = "f32"
# pnnx exits 0 and writes a model that crashes ncnn
PNNX_UNSUPPORTED_MARKER = "not supported"
SINGLE_FRAME_BATCH = 1


def onnx_input_shape(onnx_path):
    import onnx

    graph = onnx.load(onnx_path, load_external_data=False).graph
    initializer_names = {initializer.name for initializer in graph.initializer}
    graph_input = next(
        graph_input
        for graph_input in graph.input
        if graph_input.name not in initializer_names
    )
    shape = [dim.dim_value for dim in graph_input.type.tensor_type.shape.dim]
    # a named or missing dimension reads back as 0
    if 0 in shape:
        raise ValueError(f"pnnx needs a fixed input shape, {onnx_path} has {shape}")
    if shape[0] != SINGLE_FRAME_BATCH:
        raise ValueError(
            f"ncnn runs one frame at a time, {onnx_path} takes a batch of {shape[0]}"
        )
    return shape


def ncnn_model_from_onnx(onnx_path):
    param_path = converted_model_path(ENGINE_NAME, onnx_path, PARAM_SUFFIX)
    bin_path = param_path.with_suffix(BIN_SUFFIX)
    if param_path.is_file():
        return param_path

    import pnnx

    input_shape = ",".join(str(dim) for dim in onnx_input_shape(onnx_path))
    param_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=param_path.parent) as work_directory:
        work_path = Path(work_directory)
        # pnnx writes its side files beside the model it reads
        (work_path / PNNX_SOURCE_NAME).symlink_to(Path(onnx_path).resolve())
        result = subprocess.run(
            [
                pnnx.EXEC_PATH,
                PNNX_SOURCE_NAME,
                f"inputshape=[{input_shape}]{PNNX_INPUT_TYPE}",
                "fp16=0",
                f"ncnnparam={PNNX_PARAM_NAME}",
                f"ncnnbin={PNNX_BIN_NAME}",
            ],
            cwd=work_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"pnnx failed on {onnx_path}:\n{result.stdout}")
        unsupported = sorted(
            {
                line
                for line in result.stdout.splitlines()
                if PNNX_UNSUPPORTED_MARKER in line
            }
        )
        if unsupported:
            raise RuntimeError(
                f"ncnn cannot run {onnx_path}:\n" + "\n".join(unsupported)
            )
        os.replace(work_path / PNNX_BIN_NAME, bin_path)
        os.replace(work_path / PNNX_PARAM_NAME, param_path)
    return param_path


class NCNNEngine(MLEngine):
    def __init__(self):
        super().__init__()
        self.net = None
        self.model_name = None
        self.kwargs = None
        self.input_name = None
        self.output_names = None
        self._use_vulkan = False

    def do_load_model(self, model_name, **kwargs):
        """Load an NCNN model (.param + .bin files).

        model_name should be the path to the .param file.
        The .bin file is expected alongside with the same base name.
        """
        self.model_name = model_name
        self.kwargs = kwargs

        if model_name.endswith(ONNX_SUFFIX):
            model_name = str(ncnn_model_from_onnx(model_name))

        # Determine param and bin paths
        if model_name.endswith(".param"):
            param_path = model_name
            bin_path = model_name.replace(".param", ".bin")
        elif model_name.endswith(".bin"):
            bin_path = model_name
            param_path = model_name.replace(".bin", ".param")
        else:
            # Assume base name provided
            param_path = model_name + ".param"
            bin_path = model_name + ".bin"

        if not os.path.isfile(param_path):
            raise FileNotFoundError(f"NCNN param file not found: {param_path}")
        if not os.path.isfile(bin_path):
            raise FileNotFoundError(f"NCNN bin file not found: {bin_path}")

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = self._use_vulkan

        # Set thread count
        num_threads = kwargs.get("num_threads", 4)
        self.net.opt.num_threads = num_threads

        if self.net.load_param(param_path) != 0:
            raise RuntimeError(f"NCNN could not load {param_path}")
        if self.net.load_model(bin_path) != 0:
            raise RuntimeError(f"NCNN could not load {bin_path}")
        self.model = self.net
        self.input_name = self.net.input_names()[0]
        self.output_names = self.net.output_names()

        self.logger.info(
            f"NCNN model loaded: {param_path} "
            f"(vulkan: {self._use_vulkan}, threads: {num_threads})"
        )
        return True

    def do_set_device(self, device):
        """Set NCNN compute device."""
        if "vulkan" in device or "gpu" in device:
            if ncnn.get_gpu_count() == 0:
                raise RuntimeError(f"NCNN sees no Vulkan GPU for device={device}")
            self._use_vulkan = True
            self.logger.info(
                f"NCNN Vulkan GPU enabled ({ncnn.get_gpu_count()} device(s))"
            )
        elif device == "cpu":
            self._use_vulkan = False
            self.logger.info("NCNN device set to CPU")
        else:
            raise ValueError(f"Invalid device specified: {device}")
        self.device = device

        # Reload model if already loaded
        if self.model_name:
            self.do_load_model(self.model_name, **(self.kwargs or {}))

    def do_forward(self, frames):
        """Run inference using NCNN."""
        if self.net is None:
            self.logger.error("No model loaded")
            return None

        is_batch = isinstance(frames, np.ndarray) and frames.ndim == 4

        # NCNN processes one frame at a time
        if not is_batch:
            frames_list = [frames]
        else:
            frames_list = [frames[i] for i in range(frames.shape[0])]

        frame_outputs = [self._frame_outputs(frame) for frame in frames_list]
        # ncnn drops the batch axis the decoders expect
        outputs = [np.stack(per_frame) for per_frame in zip(*frame_outputs)]
        raw = outputs[0] if len(outputs) == 1 else outputs

        return self._apply_post_process(raw, is_batch)

    def _frame_outputs(self, frame):
        img = frame.astype(np.float32) / 255.0

        # NCNN expects CHW format
        if img.ndim == 3 and img.shape[2] in (1, 3, 4):
            # ncnn reads the raw buffer, a transposed view feeds it garbage
            img = np.ascontiguousarray(np.transpose(img, (2, 0, 1)))

        extractor = self.net.create_extractor()
        extractor.input(self.input_name, ncnn.Mat(img))
        outputs = []
        for output_name in self.output_names:
            ret, mat_out = extractor.extract(output_name)
            if ret != 0:
                raise RuntimeError(f"NCNN extract of {output_name} failed with {ret}")
            outputs.append(np.array(mat_out))
        return outputs

    def do_generate(self, input_text, max_length=1000, system_prompt=None):
        """NCNN does not support text generation."""
        raise NotImplementedError(
            "NCNN is a vision inference framework and does not support "
            "text generation. Use PyTorch or llama.cpp for LLM workloads."
        )
