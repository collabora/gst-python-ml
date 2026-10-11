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
import re
import time
from pathlib import Path

from .engine_factory import EngineFactory
from .ml_engine import MODEL_CACHE
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
SEGMENT_TASK = "segment"
ORIENTED_BOX_TASK = "obb"
HOST_FRAME_TASKS = (SEGMENT_TASK, ORIENTED_BOX_TASK)
TRACKER_CONFIGURATION = "botsort.yaml"
LETTERBOX_PADDING_VALUE = 114
PIXEL_MAXIMUM = 255
# LetterBox rounds this way so an odd padding pixel goes to the bottom and right
PADDING_ROUNDING_OFFSET = 0.1
HEIGHT_AND_WIDTH = 2
CONFIDENCE_COLUMN = 4
PYTORCH_WEIGHTS_FORMAT = "pt"
WARMUP_BATCH_AND_CHANNELS = (1, 3)
TENSORRT_PACKAGE = "tensorrt"
HALF_PRECISION_QUANTIZATION = 16
TENSORRT_FORMAT = "engine"
TENSORRT_SUFFIX = f".{TENSORRT_FORMAT}"
TENSORRT_EXPORT_CACHE = MODEL_CACHE / "tensorrt"
# an engine runs only on the tensorrt build and the gpu that built it
TENSORRT_EXPORT_LIBRARIES = (*YOLO_EXPORT_LIBRARIES, TENSORRT_PACKAGE)
NOT_SLUG_CHARACTERS = re.compile("[^a-z0-9]+")
SLUG_SEPARATOR = "-"


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


def tensorrt_engine_path(model_name, input_shape, device):
    import importlib.util

    import torch
    from ultralytics import YOLO

    if importlib.util.find_spec(TENSORRT_PACKAGE) is None:
        raise ImportError(
            "tensorrt=true needs the tensorrt package, install the tensorrt extra"
        )
    gpu_name = NOT_SLUG_CHARACTERS.sub(
        SLUG_SEPARATOR, torch.cuda.get_device_name(torch.device(device)).lower()
    ).strip(SLUG_SEPARATOR)
    stem = versioned_stem(model_name, TENSORRT_EXPORT_LIBRARIES)
    height, width = input_shape
    path = (
        TENSORRT_EXPORT_CACHE / f"{stem}-{gpu_name}-{height}x{width}{TENSORRT_SUFFIX}"
    )
    if path.exists():
        return str(path)
    EXPORT_WORK_DIRECTORY.mkdir(parents=True, exist_ok=True)
    TENSORRT_EXPORT_CACHE.mkdir(parents=True, exist_ok=True)
    weights = YOLO(str(EXPORT_WORK_DIRECTORY / f"{model_name}.pt"))
    exported_path = weights.export(
        format=TENSORRT_FORMAT,
        quantize=HALF_PRECISION_QUANTIZATION,
        imgsz=[height, width],
        device=torch.device(device),
        dynamic=False,
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


# the LetterBox model.predict applies to a numpy frame, auto=True for pt weights
def letterbox_layout(frame_size, imgsz, stride, pads_to_stride):
    from ultralytics.utils.checks import check_imgsz

    target_height, target_width = check_imgsz(
        imgsz, stride=stride, min_dim=HEIGHT_AND_WIDTH
    )
    height, width = frame_size
    scale = min(target_height / height, target_width / width)
    resized_height, resized_width = round(height * scale), round(width * scale)
    pad_height = target_height - resized_height
    pad_width = target_width - resized_width
    if pads_to_stride:
        pad_height, pad_width = pad_height % stride, pad_width % stride
    return (resized_height, resized_width), (pad_height, pad_width)


def letterboxed_shape(frame_size, imgsz, stride):
    (resized_height, resized_width), (pad_height, pad_width) = letterbox_layout(
        frame_size, imgsz, stride, pads_to_stride=True
    )
    return resized_height + pad_height, resized_width + pad_width


def letterboxed_device_input(frame, imgsz, stride, pads_to_stride):
    import torch

    height, width = frame.shape[:2]
    (resized_height, resized_width), (pad_height, pad_width) = letterbox_layout(
        (height, width), imgsz, stride, pads_to_stride
    )
    pad_height, pad_width = pad_height / 2, pad_width / 2
    image = frame.permute(2, 0, 1).unsqueeze(0).float() / PIXEL_MAXIMUM
    if (resized_height, resized_width) != (height, width):
        image = torch.nn.functional.interpolate(
            image,
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        )
    padding = (
        round(pad_width - PADDING_ROUNDING_OFFSET),
        round(pad_width + PADDING_ROUNDING_OFFSET),
        round(pad_height - PADDING_ROUNDING_OFFSET),
        round(pad_height + PADDING_ROUNDING_OFFSET),
    )
    return torch.nn.functional.pad(
        image, padding, value=LETTERBOX_PADDING_VALUE / PIXEL_MAXIMUM
    )


# scales in place
def frame_result(boxes, keypoints, names, model_input_shape, frame_shape):
    import numpy as np
    from ultralytics.engine.results import Results
    from ultralytics.utils import ops

    boxes[:, :BOX_COLUMNS] = ops.scale_boxes(
        model_input_shape, boxes[:, :BOX_COLUMNS], frame_shape
    )
    # results only read its shape, the frame itself stays on the device
    frame_placeholder = np.zeros(frame_shape, np.uint8)
    rescaled = Results(frame_placeholder, path=FRAME_PATH, names=names, boxes=boxes)
    if keypoints is not None:
        rescaled.update(
            keypoints=ops.scale_coords(model_input_shape, keypoints, frame_shape)
        )
    return rescaled


# ultralytics reads a tensor as rgb in 0..1 and skips its own letterbox for it
def track_device_frame(model, frame, **options):
    stride = int(model.model.stride.max())
    model_input = letterboxed_device_input(
        frame, options["imgsz"], stride, pads_to_stride=True
    )
    result = model.track(
        model_input, persist=True, tracker=TRACKER_CONFIGURATION, **options
    )[0]
    keypoints = None if result.keypoints is None else result.keypoints.data.clone()
    return frame_result(
        result.boxes.data.clone(),
        keypoints,
        result.names,
        model_input.shape[2:],
        frame.shape,
    )


# runs the model and its nms on the gpu without the ultralytics predictor
class DirectDetector:
    def __init__(self, weights, device, imgsz, pads_to_stride):
        import torch
        from ultralytics.nn.autobackend import AutoBackend
        from ultralytics.utils.checks import check_imgsz

        is_tensorrt_engine = str(weights).endswith(TENSORRT_SUFFIX)
        self.backend = AutoBackend(
            weights, device=torch.device(device), fp16=is_tensorrt_engine, verbose=False
        )
        self.runs_pytorch = self.backend.format == PYTORCH_WEIGHTS_FORMAT
        # execute_v2 runs on a stream of its own, a captured engine replays on ours
        self.waits_for_the_input = (
            self.backend.format == TENSORRT_FORMAT and self.backend.graph is None
        )
        self.pads_to_stride = pads_to_stride
        # an engine file from the user pads every frame to the size it was built for
        self.input_size = imgsz if pads_to_stride else self.backend.imgsz
        self.keypoint_shape = getattr(self.backend, "kpt_shape", None)
        warmup_height, warmup_width = check_imgsz(
            imgsz if self.runs_pytorch else self.backend.imgsz,
            stride=self.backend.stride,
            min_dim=HEIGHT_AND_WIDTH,
        )
        self.backend.warmup(
            imgsz=(*WARMUP_BATCH_AND_CHANNELS, warmup_height, warmup_width)
        )

    def detect(self, frame, confidence, iou, agnostic_nms):
        import torch

        model_input = letterboxed_device_input(
            frame,
            self.input_size,
            self.backend.stride,
            self.pads_to_stride,
        )
        with torch.inference_mode():
            if self.waits_for_the_input:
                torch.cuda.current_stream().synchronize()
            predictions = self.backend(model_input)
            if isinstance(predictions, (list, tuple)):
                predictions = predictions[0]
            detections = self._kept_detections(
                predictions[0], confidence, iou, agnostic_nms
            )
        detections = detections.float().cpu()
        keypoints = None
        if self.keypoint_shape is not None:
            keypoints = detections[:, BOX_AND_CLASS_COLUMNS:].view(
                len(detections), *self.keypoint_shape
            )
        return frame_result(
            detections[:, :BOX_AND_CLASS_COLUMNS],
            keypoints,
            self.backend.names,
            model_input.shape[2:],
            frame.shape,
        )

    # rows of x1 y1 x2 y2 confidence class then the keypoints, as non_max_suppression
    def _kept_detections(self, predictions, confidence, iou, agnostic_nms):
        import torch
        import torchvision
        from ultralytics.utils.ops import xywh2xyxy

        if self.backend.end2end:
            candidates = predictions[:, CONFIDENCE_COLUMN] > confidence
            return predictions[candidates][:MAXIMUM_DETECTIONS]
        class_count = len(self.backend.names)
        extra_count = predictions.shape[0] - BOX_COLUMNS - class_count
        boxes, class_scores, extras = predictions.transpose(0, 1).split(
            (BOX_COLUMNS, class_count, extra_count), 1
        )
        scores, classes = class_scores.max(1)
        candidates = scores > confidence
        boxes = xywh2xyxy(boxes[candidates])
        scores, classes, extras = (
            scores[candidates],
            classes[candidates],
            extras[candidates],
        )
        if agnostic_nms:
            kept = torchvision.ops.nms(boxes, scores, iou)
        else:
            kept = torchvision.ops.batched_nms(boxes, scores, classes, iou)
        kept = kept[:MAXIMUM_DETECTIONS]
        return torch.cat(
            (
                boxes[kept],
                scores[kept, None],
                classes[kept, None].to(boxes.dtype),
                extras[kept],
            ),
            1,
        )


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
        self.task_result = {
            SEGMENT_TASK: self._segmentation_result,
            ORIENTED_BOX_TASK: self._oriented_box_result,
        }.get(self.metadata["task"], self._detection_result)

    def do_forward(self, frames):
        import numpy as np

        if not isinstance(frames, np.ndarray):
            raise TypeError(
                "exported yolo models take frames in system memory, "
                "put cudadownload before the element or use engine-name=pytorch"
            )
        if self.track:
            raise ValueError(
                f"tracking runs only on the {EngineFactory.PYTORCH_ENGINE} engine"
            )
        if frames.ndim == BATCH_DIMENSIONS:
            return [self._result(frame) for frame in frames]
        return self._result(frames)

    def _result(self, frame):
        from ultralytics.data.augment import LetterBox

        model_input = LetterBox(EXPORTED_INPUT_SHAPE, auto=False)(image=frame)
        return self.task_result(frame, self.engine.do_forward(model_input))

    def _detections(self, predictions, rotated=False):
        import numpy as np
        import torch
        from ultralytics.utils import nms

        return nms.non_max_suppression(
            torch.from_numpy(np.asarray(predictions, dtype=np.float32)),
            self.conf,
            self.iou,
            agnostic=self.agnostic_nms,
            max_det=MAXIMUM_DETECTIONS,
            nc=len(self.names),
            end2end=self.end2end,
            rotated=rotated,
        )[0]

    def _detection_result(self, frame, predictions):
        from ultralytics.utils import ops

        detections = self._detections(predictions)
        detections[:, :BOX_COLUMNS] = ops.scale_boxes(
            EXPORTED_INPUT_SHAPE, detections[:, :BOX_COLUMNS], frame.shape
        )
        return self._results(frame, detections)

    def _segmentation_result(self, frame, outputs):
        import numpy as np
        import torch
        from ultralytics.engine.results import Results
        from ultralytics.utils import ops

        # ncnn returns the two outputs in the other order
        predictions, prototypes = sorted(outputs, key=np.ndim)
        detections = self._detections(predictions)
        detections[:, :BOX_COLUMNS] = ops.scale_boxes(
            EXPORTED_INPUT_SHAPE, detections[:, :BOX_COLUMNS], frame.shape
        )
        if len(detections) == 0:
            return self._results(frame, detections)
        masks = ops.process_mask_native(
            torch.from_numpy(np.asarray(prototypes, dtype=np.float32))[0],
            detections[:, BOX_AND_CLASS_COLUMNS:],
            detections[:, :BOX_COLUMNS],
            frame.shape[:2],
        )
        has_mask = masks.amax((-2, -1)) > 0
        return Results(
            frame,
            path=FRAME_PATH,
            names=self.names,
            boxes=detections[has_mask, :BOX_AND_CLASS_COLUMNS],
            masks=masks[has_mask],
        )

    def _oriented_box_result(self, frame, predictions):
        import torch
        from ultralytics.engine.results import Results
        from ultralytics.utils import ops

        detections = self._detections(predictions, rotated=True)
        rotated_boxes = torch.cat(
            [detections[:, :BOX_COLUMNS], detections[:, -1:]], dim=-1
        )
        rotated_boxes[:, :BOX_COLUMNS] = ops.scale_boxes(
            EXPORTED_INPUT_SHAPE, rotated_boxes[:, :BOX_COLUMNS], frame.shape, xywh=True
        )
        return Results(
            frame,
            path=FRAME_PATH,
            names=self.names,
            obb=torch.cat(
                [rotated_boxes, detections[:, BOX_COLUMNS:BOX_AND_CLASS_COLUMNS]],
                dim=-1,
            ),
        )

    def _results(self, frame, detections):
        from ultralytics.engine.results import Results

        return Results(
            frame,
            path=FRAME_PATH,
            names=self.names,
            boxes=detections[:, :BOX_AND_CLASS_COLUMNS],
        )


class YoloEngine(PyTorchEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tensorrt = False
        self.direct_detector = None
        self.direct_detector_frame_size = None

    def do_load_model(self, model_name, **kwargs):
        try:
            from ultralytics import YOLO

            self.model_name = model_name
            if model_name.endswith(TENSORRT_SUFFIX):
                self.model = YOLO(model_name)
            else:
                self.model = YOLO(f"{model_name}.pt")
                self.model.to(self.device)
            self.logger.info(f"YOLO model '{model_name}' loaded on {self.device}")
        except Exception as e:
            raise ValueError(f"Failed to load YOLO model '{model_name}'. Error: {e}")

    def device_frame_result(self, frame, track, **options):
        model = self.get_model()
        if model.task in HOST_FRAME_TASKS:
            raise ValueError(
                f"yolo {model.task} models take frames in system memory, "
                "put cudadownload before the element"
            )
        if track:
            return track_device_frame(model, frame, **options)
        frame_size = tuple(frame.shape[:2])
        # a tensorrt engine takes the letterboxed shape of one frame size only
        if frame_size != self.direct_detector_frame_size:
            self._build_direct_detector(frame_size, options["imgsz"])
        return self.direct_detector.detect(
            frame, options["conf"], options["iou"], options["agnostic_nms"]
        )

    def _build_direct_detector(self, frame_size, imgsz):
        if self.direct_detector is not None:
            height, width = frame_size
            self.logger.info(
                f"rebuilding the yolo detector for {width}x{height} frames"
            )
        self.direct_detector_frame_size = frame_size
        self.direct_detector = DirectDetector(
            self._direct_detector_weights(frame_size, imgsz),
            self.device,
            imgsz,
            pads_to_stride=not self.model_name.endswith(TENSORRT_SUFFIX),
        )

    def _direct_detector_weights(self, frame_size, imgsz):
        if self.model_name.endswith(TENSORRT_SUFFIX):
            return self.model_name
        model = self.get_model().model
        if not self.tensorrt:
            return model
        input_shape = letterboxed_shape(frame_size, imgsz, int(model.stride.max()))
        return tensorrt_engine_path(self.model_name, input_shape, self.device)

    def do_forward(self, frames):
        import numpy as np
        import torch

        if torch.is_tensor(frames):
            return self.device_frame_result(
                frames.to(self.device),
                self.track,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                agnostic_nms=self.agnostic_nms,
                verbose=True,
            )
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
            results = model.track(
                source=img_list,
                persist=True,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                agnostic_nms=agnostic,
                verbose=True,
                tracker=TRACKER_CONFIGURATION,
            )
        else:
            results = model(
                img_list,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                agnostic_nms=agnostic,
                verbose=True,
            )
        end_inf = time.time()

        if results is None or (isinstance(results, list) and not results):
            self.logger.warning("Inference returned None or empty list.")
            return None if not is_batch else [None] * batch_size

        self.logger.info(
            f"Preprocessing: {(end_pre - start_pre)*1000:.2f} ms, Inference: {(end_inf - end_pre)*1000:.2f} ms for {batch_size} frames"
        )
        return results[0] if not is_batch else results
