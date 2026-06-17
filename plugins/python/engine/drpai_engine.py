# DRPAIEngine
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


def _anchor_count(imgsz):
    """Total anchors a YOLO model emits for a square input at strides 8/16/32."""
    return sum((imgsz // s) ** 2 for s in (8, 16, 32))


class DRPAIEngine(MLEngine):
    """DRP-AI TVM runtime engine for Renesas RZ/V boards (RZ/V2H).

    Runs a model compiled with the Renesas DRP-AI TVM compiler on the DRP-AI
    NPU. `model_name` is the path to the compiled deploy directory containing
    ``deploy.so`` / ``deploy.json`` / ``deploy.params``.

    Inference goes through the ``drpai_runtime`` pybind11 module (built from
    ``rzv2h/`` against the board's DRP-AI TVM runtime.
    """

    def __init__(self):
        super().__init__()
        self.runtime = None
        self.model_name = None
        self.kwargs = None
        self.imgsz = 640

        self.input_format = "nchw"
        self.post_process = "anchor_free"

    def do_load_model(self, model_name, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        imgsz = kwargs.get("imgsz")
        if imgsz:
            try:
                self.imgsz = int(imgsz)
            except (TypeError, ValueError):
                pass

        try:
            import drpai_runtime
        except ImportError as e:
            self.logger.error(
                "drpai_runtime module not found. Build the pybind11 binding in "
                "rzv2h/ inside the RZ/V2H DRP-AI TVM SDK and put it on PYTHONPATH "
                f"(see rzv2h/README.md). Import error: {e}"
            )
            return False

        if not os.path.isdir(model_name):
            self.logger.error(
                f"DRP-AI model directory not found: {model_name!r} "
                "(expected a folder with deploy.so/json/params)"
            )
            return False

        try:
            self.runtime = drpai_runtime.Runtime()
            if not self.runtime.load(model_name):
                self.logger.error(f"DRP-AI failed to load model from {model_name}")
                self.runtime = None
                return False
            self.logger.info(
                f"DRP-AI model loaded from {model_name} (imgsz={self.imgsz})"
            )
            return True
        except Exception as e:
            self.logger.error(f"DRP-AI load error: {e}")
            self.runtime = None
            return False

    def do_set_device(self, device):
        self.device = device
        self.logger.info(f"DRP-AI engine device set to {device}")

    def do_generate(self, input_text, max_length=1000, system_prompt=None):
        raise NotImplementedError(
            "DRP-AI engine is a vision-inference engine; text generation is not "
            "supported."
        )

    def _preprocess(self, frame_hwc):
        """HWC uint8 RGB(A) frame -> contiguous (1, 3, H, W) float32 in [0, 1]."""
        x = np.asarray(frame_hwc, dtype=np.float32)
        if x.shape[-1] > 3:
            x = x[..., :3]
        x = x / 255.0
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)
        return np.ascontiguousarray(x, dtype=np.float32)

    def _gather_output(self):
        """Read output 0 and reshape the flat buffer to (1, 4+nc, anchors)."""
        out = np.asarray(self.runtime.get_output(0), dtype=np.float32).reshape(-1)
        anchors = _anchor_count(self.imgsz)
        if anchors and out.size % anchors == 0:
            channels = out.size // anchors
            return out.reshape(1, channels, anchors)
        self.logger.warning(
            f"DRP-AI output size {out.size} not divisible by {anchors} anchors; "
            "passing raw to post-process"
        )
        return out

    def do_forward(self, frames):
        if self.runtime is None:
            self.logger.error("DRP-AI runtime not loaded")
            return None

        is_batch = isinstance(frames, np.ndarray) and frames.ndim == 4
        batch = frames if is_batch else frames[np.newaxis, ...]

        results = []
        for img in batch:
            try:
                self.runtime.set_input(0, self._preprocess(img))
                self.runtime.run()
                raw = self._gather_output()
                results.append(self._apply_post_process(raw, is_batch=False))
            except Exception as e:
                self.logger.error(f"DRP-AI inference error: {e}")
                results.append(None)

        return results if is_batch else results[0]
