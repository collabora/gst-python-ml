# VARTEngine
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


DEVICE_NPU_ONLY = {
    "npu": False,
    "vart": False,
    "npu-only": True,
}


class VARTEngine(MLEngine):
    def __init__(self):
        super().__init__()
        self.runtime = None
        self.kwargs = None
        self.npu_only = DEVICE_NPU_ONLY["npu"]
        self.input_shapes = []
        self.input_shape_formats = []
        self.input_types = []

    def do_set_device(self, device):
        device_name = (device or "npu").lower()
        if device_name not in DEVICE_NPU_ONLY:
            choices = ", ".join(DEVICE_NPU_ONLY)
            raise ValueError(f"VART has no device={device}, use one of: {choices}")
        self.npu_only = DEVICE_NPU_ONLY[device_name]
        self.device = device_name
        self.logger.info(f"VART device set to {device_name}")
        if self.model_name:
            self.do_load_model(self.model_name, **(self.kwargs or {}))

    def do_load_model(self, model_name, **kwargs):
        snapshot_dir, separator, network_name = model_name.rpartition("::")
        if not separator:
            snapshot_dir = model_name
            network_name = ""
        if not os.path.isdir(snapshot_dir):
            raise FileNotFoundError(
                f"VART requires a snapshot directory, got: {snapshot_dir}"
            )

        try:
            from runner import VART
        except ImportError as error:
            raise ImportError(
                "VART needs the AMD Vitis AI target runtime and /etc/vai.sh"
            ) from error

        network_name = (
            kwargs.get("network_name")
            or network_name
            or os.path.basename(os.path.normpath(snapshot_dir))
        )
        output_names = kwargs.get("output_names")
        if isinstance(output_names, str):
            output_names = [name.strip() for name in output_names.split(",")]

        runtime = VART(
            snapshot_dir,
            network_name,
            output_names=output_names,
            npu_only=self.npu_only,
        )
        input_shapes = runtime.get_input_shapes()
        if len(input_shapes) != 1:
            raise ValueError(
                f"VART preliminary support requires one input, got {len(input_shapes)}"
            )

        self.runtime = runtime
        self.model = runtime
        self.model_name = model_name
        self.kwargs = kwargs
        self.input_shapes = input_shapes
        self.input_shape_formats = runtime.get_input_shape_formats()
        self.input_types = runtime.get_input_types()
        self.logger.info(
            f"VART snapshot loaded from {snapshot_dir} with network={network_name}"
        )
        return True

    def _input_format(self):
        if self.input_format != "auto":
            return self.input_format
        if not self.input_shape_formats:
            return "nhwc"
        return str(self.input_shape_formats[0]).lower()

    def do_forward(self, frames):
        if self.runtime is None:
            raise RuntimeError("VART snapshot is not loaded")

        input_array = np.asarray(frames)
        if input_array.ndim not in (3, 4):
            raise ValueError(
                "VART expects one image or a batch of images, "
                f"got shape {input_array.shape}"
            )
        is_batch = input_array.ndim == 4
        if not is_batch:
            input_array = input_array[np.newaxis, ...]
        if self._input_format() == "nchw":
            input_array = np.transpose(input_array, (0, 3, 1, 2))
        if self.input_types:
            input_array = input_array.astype(np.dtype(self.input_types[0]), copy=False)
        input_array = np.ascontiguousarray(input_array)

        outputs = self.runtime.execute([input_array])
        if not outputs:
            raise RuntimeError("VART inference returned no outputs")
        raw = outputs[0] if len(outputs) == 1 else outputs
        return self._apply_post_process(raw, is_batch=is_batch)

    def do_generate(self, input_text, max_length=1000, system_prompt=None):
        raise NotImplementedError("VART does not support text generation")
