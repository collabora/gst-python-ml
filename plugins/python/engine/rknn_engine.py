# RKNNEngine
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os

import numpy as np

from .ml_engine import MLEngine


DEVICE_CORE_MASKS = {
    "npu": None,
    "rknn": None,
    "npu:auto": None,
    "npu:0": "NPU_CORE_0",
    "npu:1": "NPU_CORE_1",
    "npu:2": "NPU_CORE_2",
    "npu:all": "NPU_CORE_ALL",
}


class RKNNEngine(MLEngine):
    def __init__(self):
        super().__init__()
        self.runtime = None
        self.kwargs = None
        self.core_mask_name = DEVICE_CORE_MASKS["npu"]

    def do_set_device(self, device):
        device_name = (device or "npu").lower()
        if device_name not in DEVICE_CORE_MASKS:
            choices = ", ".join(DEVICE_CORE_MASKS)
            raise ValueError(f"RKNN has no device={device}, use one of: {choices}")
        self.core_mask_name = DEVICE_CORE_MASKS[device_name]
        self.device = device_name
        self.logger.info(f"RKNN device set to {device_name}")
        if self.model_name:
            self.do_load_model(self.model_name, **(self.kwargs or {}))

    def _release_runtime(self):
        if self.runtime is not None:
            self.runtime.release()
        self.runtime = None
        self.model = None

    def do_load_model(self, model_name, **kwargs):
        if not os.path.isfile(model_name) or not model_name.endswith(".rknn"):
            raise FileNotFoundError(
                f"RKNN requires a .rknn model file, got: {model_name}"
            )

        try:
            from rknnlite.api import RKNNLite
        except ImportError as error:
            raise ImportError(
                "RKNN needs rknn-toolkit-lite2 on an AArch64 Rockchip board"
            ) from error

        if self.core_mask_name and not hasattr(RKNNLite, self.core_mask_name):
            raise RuntimeError(
                f"rknn-toolkit-lite2 has no {self.core_mask_name} core mask"
            )

        self._release_runtime()
        runtime = RKNNLite()
        load_status = runtime.load_rknn(model_name)
        if load_status != 0:
            runtime.release()
            raise RuntimeError(
                f"RKNN failed to load {model_name} with status {load_status}"
            )

        if self.core_mask_name:
            core_mask = getattr(RKNNLite, self.core_mask_name)
            initialization_status = runtime.init_runtime(core_mask=core_mask)
        else:
            initialization_status = runtime.init_runtime()
        if initialization_status != 0:
            runtime.release()
            raise RuntimeError(
                "RKNN failed to initialize the NPU "
                f"with status {initialization_status}"
            )

        self.runtime = runtime
        self.model = runtime
        self.model_name = model_name
        self.kwargs = kwargs
        self.logger.info(
            f"RKNN model loaded from {model_name} with device={self.device or 'npu'}"
        )
        return True

    def _prepare_frame(self, frame):
        frame_array = np.asarray(frame)
        if frame_array.ndim != 3:
            raise ValueError(
                f"RKNN expects an HWC image frame, got shape {frame_array.shape}"
            )
        if self.input_format == "nchw":
            input_array = np.transpose(frame_array, (2, 0, 1))[np.newaxis, ...]
            data_format = "nchw"
        else:
            input_array = frame_array[np.newaxis, ...]
            data_format = "nhwc"
        return np.ascontiguousarray(input_array), data_format

    def _forward_frame(self, frame):
        input_array, data_format = self._prepare_frame(frame)
        outputs = self.runtime.inference(
            inputs=[input_array], data_format=[data_format]
        )
        if outputs is None:
            raise RuntimeError("RKNN inference returned no outputs")
        single_output = isinstance(outputs, (list, tuple)) and len(outputs) == 1
        raw = outputs[0] if single_output else outputs
        return self._apply_post_process(raw, is_batch=False)

    def do_forward(self, frames):
        if self.runtime is None:
            raise RuntimeError("RKNN model is not loaded")
        frame_array = np.asarray(frames)
        if frame_array.ndim == 3:
            return self._forward_frame(frame_array)
        if frame_array.ndim == 4:
            return [self._forward_frame(frame) for frame in frame_array]
        raise ValueError(
            f"RKNN expects one image or a batch of images, got shape {frame_array.shape}"
        )

    def do_generate(self, input_text, max_length=1000, system_prompt=None):
        raise NotImplementedError("RKNN does not support text generation")
