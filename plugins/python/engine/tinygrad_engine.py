# TinyGradEngine
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
import os
from pathlib import Path

import numpy as np

from .ml_engine import MLEngine, TORCHVISION_WEIGHTS, is_torchvision_resnet

# tinygrad loads the nvrtc library this names
NVRTC_PATH_VARIABLE = "NVRTC_PATH"
LIBRARY_PATH_VARIABLE = "LD_LIBRARY_PATH"


class TinyGradEngine(MLEngine):
    def __init__(self):
        super().__init__()
        self.model_type = None
        self.model_name = None
        self.kwargs = None

    def do_load_model(self, model_name, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

        if os.path.isfile(model_name) and model_name.endswith(".onnx"):
            from tinygrad import TinyJit
            from tinygrad.nn.onnx import OnnxRunner

            runner = OnnxRunner(model_name)
            self.model = runner
            # every frame takes about a second without the jit
            self.run_onnx = TinyJit(
                lambda **inputs: [
                    output.realize() for output in runner(inputs).values()
                ]
            )
            self.model_type = "custom"
            self.logger.info(f"ONNX model loaded with TinyGrad: {model_name}")
            return True

        # TorchVision models
        from torchvision import models as tv_models

        if hasattr(tv_models, model_name):
            if not is_torchvision_resnet(model_name):
                raise ValueError(
                    f"TinyGrad runs the torchvision resnet family, not '{model_name}'."
                )
            pt_model = getattr(tv_models, model_name)(weights=TORCHVISION_WEIGHTS)
            from .tinygrad_resnet import tinygrad_resnet

            self.model = tinygrad_resnet(pt_model.eval())
            self.model_type = "classification"
            self.logger.info(
                f"Pre-trained vision model '{model_name}' loaded with TinyGrad."
            )
            return True

        raise FileNotFoundError(
            f"TinyGrad takes a .onnx file or a torchvision resnet name, got: {model_name}"
        )

    def do_set_device(self, device):
        """Set TinyGrad device for the model."""
        from tinygrad import Device
        from tinygrad.helpers import DEV

        device_upper = device.upper() if device else "CPU"
        if device_upper == "CUDA":
            self._use_the_nvrtc_torch_ships()
        # raises when tinygrad cannot open the device
        Device[device_upper]
        DEV.value = device_upper
        self.device = device
        self.logger.info(f"Setting device to {device}")

    # a system nvrtc newer than the driver fails with CUDA_ERROR_UNSUPPORTED_PTX_VERSION
    def _use_the_nvrtc_torch_ships(self):
        if os.environ.get(NVRTC_PATH_VARIABLE):
            return
        import nvidia
        import torch

        if torch.version.cuda is None:
            return
        cuda_major = torch.version.cuda.split(".")[0]
        libraries = [
            library
            for package_path in nvidia.__path__
            for library in Path(package_path).glob(f"*/lib/libnvrtc.so.{cuda_major}")
        ]
        if not libraries:
            return
        nvrtc = libraries[0]
        # nvrtc does not look for its builtins beside itself
        for builtins in nvrtc.parent.glob("libnvrtc-builtins.so.*"):
            ctypes.CDLL(str(builtins), mode=ctypes.RTLD_GLOBAL)
        # tinygrad's compile workers only inherit the environment
        os.environ[LIBRARY_PATH_VARIABLE] = os.pathsep.join(
            filter(None, [str(nvrtc.parent), os.environ.get(LIBRARY_PATH_VARIABLE)])
        )
        os.environ[NVRTC_PATH_VARIABLE] = str(nvrtc)
        self.logger.info(f"TinyGrad compiles its CUDA kernels with {nvrtc}")

    def _forward_classification(self, frames):
        """Handle inference for classification models."""

        is_batch = frames.ndim == 4
        img_array = np.array(frames, dtype=np.float32) / 255.0
        if is_batch:
            img_array = np.transpose(img_array, (0, 3, 1, 2))
        else:
            img_array = np.transpose(img_array, (2, 0, 1))
            img_array = np.expand_dims(img_array, 0)

        from tinygrad import Tensor

        preds = self.model(Tensor(img_array)).numpy()
        probs = np.exp(preds) / np.sum(np.exp(preds), axis=1, keepdims=True)
        top_classes = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        results = [
            {"labels": [int(c)], "scores": [float(s)]}
            for c, s in zip(top_classes, confidences)
        ]
        return results[0] if not is_batch else results

    def do_forward(self, frames):
        """Execute inference on a single frame or batch of frames."""
        is_batch = isinstance(frames, np.ndarray) and frames.ndim == 4
        if not isinstance(frames, np.ndarray):
            self.logger.error(f"Invalid input type for forward: {type(frames)}")
            return None

        if self.model_type == "classification":
            return self._forward_classification(frames)

        elif self.model_type == "custom":
            return self._forward_onnx(frames, is_batch)

        else:
            raise ValueError("Unsupported model type.")

    def _forward_onnx(self, frames, is_batch):
        from tinygrad import Tensor

        input_name, input_value = next(iter(self.model.graph_inputs.items()))
        input_shape = input_value.shape
        if self.input_format == "auto" and len(input_shape) == 4:
            self.input_format = "nchw" if input_shape[1] in (1, 3, 4) else "nhwc"
        img = self._apply_input_format(frames.astype(np.float32) / 255.0, is_batch)
        outputs = self.run_onnx(
            **{input_name: Tensor(np.ascontiguousarray(img)).realize()}
        )
        arrays = [output.numpy() for output in outputs]
        raw = arrays if len(arrays) > 1 else arrays[0]
        return self._apply_post_process(raw, is_batch)

    def do_generate(self, input_text, max_length=1000, system_prompt=None):
        raise NotImplementedError(
            "TinyGrad does not support text generation. "
            "Use PyTorch or llama.cpp for LLM workloads."
        )
