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

import ast
import time
from pathlib import Path

from .engine_factory import EngineFactory
from .onnx_export import ONNX_EXPORT_CACHE, executorch_export_path, versioned_stem
from .pytorch_engine import PyTorchEngine

EXPORTED_INPUT_SIZE = 640
EXPORTED_INPUT_SHAPE = (EXPORTED_INPUT_SIZE, EXPORTED_INPUT_SIZE)
# an interrupted export must not reach the cache
EXPORT_WORK_DIRECTORY = ONNX_EXPORT_CACHE / "ultralytics"
YOLO_EXPORT_LIBRARIES = ("torch", "ultralytics")
BATCH_DIMENSIONS = 4
ULTRALYTICS_EXECUTORCH_FILE = "model.pte"
# the ultralytics defaults the pytorch pose engine runs with
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.7
MAXIMUM_DETECTIONS = 300
BOX_COLUMNS = 4
BOX_AND_CLASS_COLUMNS = 6
# a pipeline frame has no file behind it
FRAME_PATH = ""


def exported_yolo_path(model_name):
    import torch
    from ultralytics import YOLO

    path = (
        ONNX_EXPORT_CACHE / f"{versioned_stem(model_name, YOLO_EXPORT_LIBRARIES)}.onnx"
    )
    if path.exists():
        return str(path)
    EXPORT_WORK_DIRECTORY.mkdir(parents=True, exist_ok=True)
    weights = YOLO(str(EXPORT_WORK_DIRECTORY / f"{model_name}.pt"))
    # a device name makes ultralytics hide the gpu from the whole process
    exported_path = weights.export(
        format="onnx", imgsz=EXPORTED_INPUT_SIZE, device=torch.device("cpu")
    )
    Path(exported_path).rename(path)
    return str(path)


def exported_yolo_executorch_path(model_name):
    import torch
    from ultralytics import YOLO

    path = executorch_export_path(versioned_stem(model_name, YOLO_EXPORT_LIBRARIES))
    if path.exists():
        return str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = YOLO(str(EXPORT_WORK_DIRECTORY / f"{model_name}.pt"))
    exported_directory = weights.export(
        format="executorch", imgsz=EXPORTED_INPUT_SIZE, device=torch.device("cpu")
    )
    Path(exported_directory, ULTRALYTICS_EXECUTORCH_FILE).rename(path)
    return str(path)


def exported_yolo_metadata(path):
    import onnx

    return {entry.key: entry.value for entry in onnx.load(path).metadata_props}


class ExportedYolo:
    def __init__(self, model_name, engine_name):
        self.engine = None
        self.track = False
        self.conf = DEFAULT_CONFIDENCE
        self.iou = DEFAULT_IOU
        self.agnostic_nms = False
        onnx_path = exported_yolo_path(model_name)
        self.metadata = exported_yolo_metadata(onnx_path)
        self.path = onnx_path
        if engine_name == EngineFactory.EXECUTORCH_ENGINE:
            self.path = exported_yolo_executorch_path(model_name)
        self.names = ast.literal_eval(self.metadata["names"])
        self.end2end = ast.literal_eval(self.metadata["end2end"])

    def do_forward(self, frames):
        if self.track:
            raise ValueError(
                f"tracking runs only on the {EngineFactory.PYTORCH_ENGINE} engine"
            )
        if frames.ndim == BATCH_DIMENSIONS:
            return [self._result(frame) for frame in frames]
        return self._result(frames)

    def _result(self, frame):
        import numpy as np
        import torch
        from ultralytics.data.augment import LetterBox
        from ultralytics.utils import nms, ops

        model_input = LetterBox(EXPORTED_INPUT_SHAPE, auto=False)(image=frame)
        output = np.asarray(self.engine.do_forward(model_input), dtype=np.float32)
        detections = nms.non_max_suppression(
            torch.from_numpy(output),
            self.conf,
            self.iou,
            agnostic=self.agnostic_nms,
            max_det=MAXIMUM_DETECTIONS,
            nc=len(self.names),
            end2end=self.end2end,
        )[0]
        detections[:, :BOX_COLUMNS] = ops.scale_boxes(
            EXPORTED_INPUT_SHAPE, detections[:, :BOX_COLUMNS], frame.shape
        )
        return self._results(frame, detections)

    def _results(self, frame, detections):
        from ultralytics.engine.results import Results

        return Results(
            frame,
            path=FRAME_PATH,
            names=self.names,
            boxes=detections[:, :BOX_AND_CLASS_COLUMNS],
        )


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
        # ultralytics reads a numpy frame as bgr
        writable_frames = np.ascontiguousarray(frames[..., ::-1])
        batch_size = writable_frames.shape[0] if is_batch else 1

        model = self.get_model()
        if model is None:
            self.logger.error("Model is not loaded.")
            return None if not is_batch else [None] * batch_size

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
        imgsz = getattr(self, "imgsz", 640)
        if self.track:
            # Ensure tracker persists across batches
            results = self.execute_with_stream(
                lambda: model.track(
                    source=img_list,
                    persist=True,
                    imgsz=imgsz,
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
                    imgsz=imgsz,
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
