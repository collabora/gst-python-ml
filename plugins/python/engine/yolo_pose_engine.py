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

from .pytorch_engine import PyTorchEngine


class YoloPoseEngine(PyTorchEngine):
    """PyTorch engine for YOLO pose estimation models."""

    def do_load_model(self, model_name, **kwargs):
        try:
            from ultralytics import YOLO

            self.model = YOLO(f"{model_name}.pt")
            self.execute_with_stream(lambda: self.model.to(self.device))
            self.logger.info(f"YOLO pose model '{model_name}' loaded on {self.device}")
        except Exception as e:
            raise ValueError(f"Failed to load YOLO pose model '{model_name}': {e}")

    def do_forward(self, frames):
        import numpy as np

        is_batch = isinstance(frames, np.ndarray) and frames.ndim == 4
        writable = np.array(frames, copy=True)
        batch_size = writable.shape[0] if is_batch else 1

        model = self.get_model()
        if model is None:
            self.logger.error("Pose model not loaded")
            return None if not is_batch else [None] * batch_size

        img_list = [writable[i] for i in range(batch_size)] if is_batch else [writable]
        results = self.execute_with_stream(
            lambda: model(img_list, imgsz=640, conf=0.25, verbose=False)
        )
        if not results:
            return None if not is_batch else [None] * batch_size
        return results[0] if not is_batch else results
