# YoloEngine
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

import time

from .pytorch_engine import PyTorchEngine


class YoloEngine(PyTorchEngine):
    def do_load_model(self, model_name, **kwargs):
        try:
            from ultralytics import YOLO

            self.model = YOLO(f"{model_name}.pt")
            self.execute_with_stream(lambda: self.model.to(self.device))
            self.logger.info(f"YOLO model '{model_name}' loaded on {self.device}")
        except Exception as e:
            raise ValueError(f"Failed to load YOLO model '{model_name}'. Error: {e}")

    def do_forward(self, frames):
        import numpy as np

        is_batch = isinstance(frames, np.ndarray) and frames.ndim == 4
        writable_frames = np.array(frames, copy=True)
        batch_size = writable_frames.shape[0] if is_batch else 1

        model = self.get_model()
        if model is None:
            self.logger.error("Model is not loaded.")
            return None if not is_batch else [None] * batch_size

        try:
            start_pre = time.time()
            img_list = (
                [
                    writable_frames[i] if is_batch else writable_frames
                    for i in range(batch_size)
                ]
                if is_batch
                else [writable_frames]
            )
            self.logger.debug(
                f"Input shape: {writable_frames.shape}, min={writable_frames.min()}, max={writable_frames.max()}"
            )
            end_pre = time.time()

            conf = getattr(self, "conf", 0.25)
            iou = getattr(self, "iou", 0.5)
            agnostic = getattr(self, "agnostic_nms", True)
            if self.track:
                # Ensure tracker persists across batches
                results = self.execute_with_stream(
                    lambda: model.track(
                        source=img_list,
                        persist=True,
                        imgsz=640,
                        conf=conf,
                        iou=iou,
                        agnostic_nms=agnostic,
                        verbose=True,
                        tracker="botsort.yaml",
                    )
                )
            else:
                results = self.execute_with_stream(
                    lambda: model(
                        img_list,
                        imgsz=640,
                        conf=conf,
                        iou=iou,
                        agnostic_nms=agnostic,
                        verbose=True,
                    )
                )
            end_inf = time.time()

            if results is None or (isinstance(results, list) and not results):
                self.logger.warning("Inference returned None or empty list.")
                return None if not is_batch else [None] * batch_size

            self.logger.info(
                f"Preprocessing: {(end_pre - start_pre)*1000:.2f} ms, Inference: {(end_inf - end_pre)*1000:.2f} ms for {batch_size} frames"
            )
            return results[0] if not is_batch else results

        except Exception as e:
            self.logger.error(f"Error during inference: {e}")
            return None if not is_batch else [None] * batch_size
