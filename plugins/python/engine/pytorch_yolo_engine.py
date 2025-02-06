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
        """
        Override base method to load the YOLO model (detection or segmentation).
        """
        try:
            self.set_model(YOLO(f"{model_name}.pt"))
            self.execute_with_stream(lambda: self.model.to(self.device))
            self.logger.info(
                f"YOLO model '{model_name}' loaded successfully on {self.device}"
            )
        except Exception as e:
            raise ValueError(f"Failed to load YOLO model '{model_name}'. Error: {e}")

    def forward(self, frame):
        """
        Perform inference using the YOLO model.
        """
        # Make a writable copy of the frame to avoid non-writable tensor warnings
        writable_frame = np.array(frame, copy=True)

        # Ensure model is loaded before attempting inference
        model = self.get_model()
        if model is None:
            self.logger.error("forward: Model is not loaded.")
            return []

        try:
            # Perform tracking or regular inference
            if self.track:
                results = self.execute_with_stream(
                    lambda: model.track(source=writable_frame, persist=True)
                )
            else:
                results = self.execute_with_stream(lambda: model([writable_frame]))

            # Log and handle None results explicitly
            if results is None:
                self.logger.warning(
                    "forward: Inference returned None; defaulting to an empty list."
                )
                return []

            self.logger.debug(f"forward: Inference results received: {results}")
            return results

        except Exception as e:
            self.logger.error(f"forward: Error during inference: {e}")
            return []
