# YoloPoseEngine
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

import ast

from .yolo_engine import (
    BOX_AND_CLASS_COLUMNS,
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU,
    EXPORTED_INPUT_SHAPE,
    ExportedYolo,
    YoloEngine,
)

POSE_INPUT_SIZE = 640
POSE_PREDICTION_OPTIONS = {
    "imgsz": POSE_INPUT_SIZE,
    "conf": DEFAULT_CONFIDENCE,
    "iou": DEFAULT_IOU,
    "agnostic_nms": False,
    "verbose": False,
}


class ExportedYoloPose(ExportedYolo):
    def __init__(self, model_name, engine_name):
        super().__init__(model_name, engine_name)
        self.keypoint_shape = ast.literal_eval(self.metadata["kpt_shape"])

    def _results(self, frame, detections):
        from ultralytics.utils import ops

        results = super()._results(frame, detections)
        keypoints = detections[:, BOX_AND_CLASS_COLUMNS:].view(
            len(detections), *self.keypoint_shape
        )
        results.update(
            keypoints=ops.scale_coords(EXPORTED_INPUT_SHAPE, keypoints, frame.shape)
        )
        return results


class YoloPoseEngine(YoloEngine):
    """PyTorch engine for YOLO pose estimation models."""

    def do_forward(self, frames):
        import numpy as np
        import torch

        if torch.is_tensor(frames):
            return self.device_frame_result(
                frames.to(self.device), track=False, **POSE_PREDICTION_OPTIONS
            )
        is_batch = isinstance(frames, np.ndarray) and frames.ndim == 4
        # ultralytics reads a numpy frame as bgr
        writable = np.ascontiguousarray(frames[..., ::-1])
        batch_size = writable.shape[0] if is_batch else 1

        model = self.get_model()
        if model is None:
            self.logger.error("Pose model not loaded")
            return None if not is_batch else [None] * batch_size

        img_list = [writable[i] for i in range(batch_size)] if is_batch else [writable]
        results = model(img_list, **POSE_PREDICTION_OPTIONS)
        if not results:
            return None if not is_batch else [None] * batch_size
        return results[0] if not is_batch else results
