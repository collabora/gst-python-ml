# PyTorchYoloEngine
import numpy as np
import time
from ultralytics import YOLO
from .pytorch_engine import PyTorchEngine


class PyTorchYoloEngine(PyTorchEngine):
    def load_model(self, model_name, **kwargs):
        try:
            self.set_model(YOLO(f"{model_name}.pt"))
            self.execute_with_stream(lambda: self.model.to(self.device))
            self.logger.info(f"YOLO model '{model_name}' loaded on {self.device}")
        except Exception as e:
            raise ValueError(f"Failed to load YOLO model '{model_name}'. Error: {e}")

    def forward(self, frames):
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

            if self.track:
                # Ensure tracker persists across batches
                results = self.execute_with_stream(
                    lambda: model.track(
                        source=img_list,
                        persist=True,
                        imgsz=640,
                        conf=0.1,
                        verbose=True,
                        tracker="botsort.yaml",
                    )
                )
            else:
                results = self.execute_with_stream(
                    lambda: model(img_list, imgsz=640, conf=0.1, verbose=True)
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
