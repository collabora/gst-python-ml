# PyTorchYoloEngine
# Copyright (C) 2024-2025 Collabora Ltd.
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

import numpy as np
from ultralytics import YOLO
from .pytorch_engine import PyTorchEngine


class PyTorchYoloEngine(PyTorchEngine):
    def load_model(self, model_name, **kwargs):
        """Load the YOLO model (detection or segmentation)."""
        try:
            self.set_model(YOLO(f"{model_name}.pt"))
            self.execute_with_stream(lambda: self.model.to(self.device))
            self.logger.info(
                f"YOLO model '{model_name}' loaded successfully on {self.device}"
            )
        except Exception as e:
            raise ValueError(f"Failed to load YOLO model '{model_name}'. Error: {e}")

    def forward(self, frames):
        """Perform inference using the YOLO model, supporting single frames or batches."""
        is_batch = isinstance(frames, np.ndarray) and frames.ndim == 4  # (B, H, W, C)
        writable_frames = np.array(frames, copy=True)
        if not is_batch:
            writable_frames = [writable_frames]  # YOLO expects a list for single frame
        else:
            writable_frames = list(writable_frames)  # Convert batch to list of frames

        model = self.get_model()
        if model is None:
            self.logger.error("forward: Model is not loaded.")
            return [] if is_batch else None

        try:
            if self.track:
                results = self.execute_with_stream(
                    lambda: model.track(source=writable_frames, persist=True)
                )
            else:
                results = self.execute_with_stream(lambda: model(writable_frames))

            if results is None:
                self.logger.warning(
                    "Inference returned None; defaulting to empty results."
                )
                return [] if is_batch else None

            self.logger.debug(f"forward: Inference results received: {results}")
            return (
                results if is_batch else results[0]
            )  # List for batch, single result otherwise

        except Exception as e:
            self.logger.error(f"forward: Error during inference: {e}")
            return [] if is_batch else None
