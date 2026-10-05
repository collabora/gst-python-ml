# TVMEngine
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
import tvm
from tvm import relax

from .ml_engine import MLEngine

CLASSIFIER_INPUT_SHAPE = (1, 3, 224, 224)
# the torch importer names the exported graph main
ENTRY_FUNCTION = "main"


class TVMEngine(MLEngine):
    def __init__(self):
        super().__init__()
        self.vm = None
        self.model_type = None
        self.model_name = None
        self.kwargs = None
        self.tvm_device = None

    def do_load_model(self, model_name, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

        if os.path.isfile(model_name) and model_name.endswith((".so", ".tar")):
            self._start_vm(tvm.runtime.load_module(model_name))
            self.model_type = "custom"
            self.logger.info(f"TVM compiled model loaded from local path: {model_name}")
            return True

        from torchvision import models as tv_models

        if hasattr(tv_models, model_name):
            pt_model = getattr(tv_models, model_name)(pretrained=True)
            self._compile_pytorch_model(pt_model, CLASSIFIER_INPUT_SHAPE)
            self.model_type = "classification"
            self.logger.info(
                f"Pre-trained vision model '{model_name}' compiled with TVM."
            )
            return True

        raise FileNotFoundError(
            f"TVM takes a compiled .so or .tar or a torchvision model name, got: {model_name}"
        )

    def _compile_pytorch_model(self, pt_model, input_shape):
        import torch
        from tvm.relax.frontend.torch import from_exported_program

        pt_model.eval()
        exported = torch.export.export(pt_model, (torch.randn(*input_shape),))
        module = from_exported_program(exported)
        self._start_vm(tvm.compile(module, self._get_target()))

    def _start_vm(self, module):
        self.tvm_device = self._get_tvm_device()
        self.vm = relax.VirtualMachine(module, self.tvm_device)

    def _run(self, img):
        input_tensor = tvm.runtime.tensor(np.ascontiguousarray(img), self.tvm_device)
        outputs = self.vm[ENTRY_FUNCTION](input_tensor)
        if not hasattr(outputs, "__len__"):
            outputs = [outputs]
        return [output.numpy() for output in outputs]

    def _get_target(self):
        if self.device and "cuda" in self.device:
            return tvm.target.Target("cuda", host="llvm")
        return tvm.target.Target("llvm")

    def _get_tvm_device(self):
        if self.device and "cuda" in self.device:
            index = 0
            if ":" in self.device:
                try:
                    index = int(self.device.split(":")[-1])
                except ValueError:
                    pass
            return tvm.cuda(index)
        return tvm.cpu(0)

    def do_set_device(self, device):
        if "cuda" in device and not tvm.cuda(0).exist:
            raise RuntimeError(f"TVM sees no CUDA device for device={device}")
        self.device = device
        self.logger.info(f"Setting device to {device}")

        if self.model_name:
            self.do_load_model(self.model_name, **self.kwargs)

    def _forward_classification(self, frames):
        is_batch = frames.ndim == 4
        img_array = np.array(frames, dtype=np.float32) / 255.0
        if is_batch:
            img_array = np.transpose(img_array, (0, 3, 1, 2))
        else:
            img_array = np.transpose(img_array, (2, 0, 1))
            img_array = np.expand_dims(img_array, 0)

        preds = self._run(img_array)[0]
        probs = np.exp(preds) / np.sum(np.exp(preds), axis=1, keepdims=True)
        top_classes = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        results = [
            {"labels": [int(c)], "scores": [float(s)]}
            for c, s in zip(top_classes, confidences)
        ]
        return results[0] if not is_batch else results

    def do_forward(self, frames):
        is_batch = isinstance(frames, np.ndarray) and frames.ndim == 4
        if not isinstance(frames, np.ndarray):
            self.logger.error(f"Invalid input type for forward: {type(frames)}")
            return None

        if self.model_type == "classification":
            return self._forward_classification(frames)

        elif self.model_type == "custom":
            img = self._apply_input_format(frames.astype(np.float32) / 255.0, is_batch)
            outputs = self._run(img)
            raw = outputs if len(outputs) > 1 else outputs[0]
            return self._apply_post_process(raw, is_batch)

        else:
            raise ValueError("Unsupported model type.")

    def do_generate(self, input_text, max_length=1000, system_prompt=None):
        raise NotImplementedError(
            "TVM does not support text generation. "
            "Use PyTorch or llama.cpp for LLM workloads."
        )
