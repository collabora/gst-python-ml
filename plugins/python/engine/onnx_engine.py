# ONNXEngine
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

import ctypes
import importlib.util
import os
import tempfile
from pathlib import Path
import numpy as np
import onnxruntime as ort

from .ml_engine import MLEngine

TENSORRT_PROVIDER = "TensorrtExecutionProvider"
# providers are tried in the order listed
DEVICE_PROVIDERS = {
    "tensorrt": (TENSORRT_PROVIDER,),
    "cuda": ("CUDAExecutionProvider",),
    "rocm": ("MIGraphXExecutionProvider", "ROCMExecutionProvider"),
    "hip": ("MIGraphXExecutionProvider", "ROCMExecutionProvider"),
    "npu": ("VitisAIExecutionProvider",),
    "ryzenai": ("VitisAIExecutionProvider",),
}
PROVIDERS_WITHOUT_DEVICE_ID = ("VitisAIExecutionProvider",)
PROVIDERS_ON_NVIDIA_LIBRARIES = (TENSORRT_PROVIDER, "CUDAExecutionProvider")
# an uncached engine takes minutes to build on every start
TENSORRT_ENGINE_CACHE = Path.home() / ".cache" / "gst-python-ml" / "tensorrt"
# device=tensorrt-fp16 builds a half precision engine
TENSORRT_HALF_PRECISION_SUFFIX = "fp16"
TENSORRT_LIBRARIES_PACKAGE = "tensorrt_libs"
TENSORRT_LIBRARY_PATTERNS = (
    "libnvinfer.so.*",
    "libnvinfer_plugin.so.*",
    "libnvonnxparser.so.*",
)


class ONNXEngine(MLEngine):
    def __init__(self):
        super().__init__()
        self.model = None
        self.session = None
        self.model_type = None
        self.model_name = None
        self.kwargs = None
        self.provider = "CPUExecutionProvider"
        self.input_names = None
        self.output_names = None

    def _providers(self):
        """Return provider list with CPU fallback."""
        if self.provider == "CPUExecutionProvider":
            return ["CPUExecutionProvider"]
        return [self.provider, "CPUExecutionProvider"]

    def _create_session(self, model_path):
        session = ort.InferenceSession(model_path, providers=self._providers())
        provider = self.provider if isinstance(self.provider, str) else self.provider[0]
        # onnxruntime drops a provider whose libraries fail to load
        if provider not in session.get_providers():
            raise RuntimeError(
                f"onnxruntime could not start {provider} for device={self.device}"
            )
        return session

    def _input_is_nchw(self):
        """Auto-detect whether the model's first input expects NCHW layout."""
        if self.session is None:
            return False
        shape = self.session.get_inputs()[0].shape
        return len(shape) == 4 and shape[1] in (1, 3, 4)

    def _model_input_hw(self):
        """(H, W) the model's input expects, or None if dynamic/unknown."""
        if self.session is None:
            return None
        shape = self.session.get_inputs()[0].shape
        if len(shape) != 4:
            return None
        h, w = shape[2], shape[3]
        if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
            return (h, w)
        return None

    def _letterbox(self, frames, is_batch):
        """Resize frame(s) to the model input size, preserving aspect ratio with
        grey padding (YOLO-style). Returns (processed, transform); transform =
        (ratio, pad_x, pad_y, orig_w, orig_h) maps model coords back to the
        original frame. Returns (frames, None) when no resize is needed (already
        model-sized, or dynamic input) -- so pre-sized callers are unaffected."""
        import numpy as np
        import cv2

        mhw = self._model_input_hw()
        if mhw is None:
            return frames, None
        mh, mw = mhw
        imgs = frames if is_batch else frames[None]
        h, w = int(imgs.shape[1]), int(imgs.shape[2])
        if (h, w) == (mh, mw):
            return frames, None
        r = min(mh / h, mw / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        pad_x, pad_y = (mw - nw) // 2, (mh - nh) // 2
        out = np.full((imgs.shape[0], mh, mw, imgs.shape[3]), 114, dtype=imgs.dtype)
        for i in range(imgs.shape[0]):
            out[i, pad_y : pad_y + nh, pad_x : pad_x + nw] = cv2.resize(
                imgs[i], (nw, nh), interpolation=cv2.INTER_LINEAR
            )
        proc = out if is_batch else out[0]
        return proc, (r, float(pad_x), float(pad_y), w, h)

    def _unletterbox(self, results, transform):
        """Map detection boxes from model coords back to original-frame coords."""
        import numpy as np

        r, pad_x, pad_y, ow, oh = transform
        for res in results if isinstance(results, list) else [results]:
            if not isinstance(res, dict):
                continue
            b = res.get("boxes")
            if b is None or len(b) == 0:
                continue
            b = np.asarray(b, dtype=np.float32).copy()
            b[:, [0, 2]] = ((b[:, [0, 2]] - pad_x) / r).clip(0, ow)
            b[:, [1, 3]] = ((b[:, [1, 3]] - pad_y) / r).clip(0, oh)
            res["boxes"] = b

    def do_load_model(self, model_name, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

        if os.path.isfile(model_name) and model_name.endswith(".onnx"):
            self.session = self._create_session(model_name)
            self.model = self.session
            self.model_type = "custom"
            self.input_names = [inp.name for inp in self.session.get_inputs()]
            self.output_names = [out.name for out in self.session.get_outputs()]
            self.logger.info(
                f"ONNX model loaded from local path: {model_name} "
                f"(active providers: {self.session.get_providers()})"
            )
            return True
        else:
            from torchvision import models as tv_models

            if hasattr(tv_models, model_name):
                pt_model = getattr(tv_models, model_name)(pretrained=True)
                self.model_type = "classification"
            elif hasattr(tv_models.detection, model_name):
                pt_model = getattr(tv_models.detection, model_name)(pretrained=True)
                self.model_type = "detection"
            else:
                raise FileNotFoundError(
                    f"ONNX takes a .onnx file or a torchvision model name, got: {model_name}"
                )

            # For TorchVision models, export to ONNX
            import torch

            pt_model.eval()
            dummy_input = torch.randn(1, 3, 224, 224)
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp_file:
                torch.onnx.export(
                    pt_model,
                    dummy_input,
                    tmp_file.name,
                    opset_version=11,
                    input_names=["input"],
                    output_names=(
                        ["output"]
                        if self.model_type == "classification"
                        else ["boxes", "labels", "scores"]
                    ),
                    dynamic_axes=(
                        {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
                        if self.model_type == "classification"
                        else None
                    ),
                )
                self.session = self._create_session(tmp_file.name)
            os.unlink(tmp_file.name)
            self.model = self.session
            self.input_names = [inp.name for inp in self.session.get_inputs()]
            self.output_names = [out.name for out in self.session.get_outputs()]
            self.logger.info(
                f"Pre-trained model '{model_name}' exported to ONNX and loaded."
            )

        return True

    def do_set_device(self, device):
        """Set ONNX device for the model."""
        self.provider = self._provider_for(device)
        self.device = device
        self.logger.info(f"Setting device to {device}")

        # Reload model if already loaded
        if self.model_name:
            self.do_load_model(self.model_name, **self.kwargs)

    def _provider_for(self, device):
        if device == "cpu":
            return "CPUExecutionProvider"
        wanted = next(
            (
                providers
                for keyword, providers in DEVICE_PROVIDERS.items()
                if keyword in device
            ),
            None,
        )
        if wanted is None:
            raise ValueError(f"Invalid device specified: {device}")
        available = ort.get_available_providers()
        provider = next((name for name in wanted if name in available), None)
        if provider is None:
            raise RuntimeError(
                f"device={device} needs {' or '.join(wanted)}, "
                f"this onnxruntime has {', '.join(available)}"
            )
        if provider in PROVIDERS_WITHOUT_DEVICE_ID:
            return provider
        if provider in PROVIDERS_ON_NVIDIA_LIBRARIES:
            # cuda and cudnn come in pip wheels the loader does not search
            ort.preload_dlls()
        options = {"device_id": int(device.split(":")[-1]) if ":" in device else 0}
        if provider == TENSORRT_PROVIDER:
            self._preload_tensorrt_libraries()
            TENSORRT_ENGINE_CACHE.mkdir(parents=True, exist_ok=True)
            options["trt_engine_cache_enable"] = True
            options["trt_engine_cache_path"] = str(TENSORRT_ENGINE_CACHE)
            options["trt_fp16_enable"] = TENSORRT_HALF_PRECISION_SUFFIX in device
        return (provider, options)

    # the pip wheel's tensorrt is not on the library path
    def _preload_tensorrt_libraries(self):
        package = importlib.util.find_spec(TENSORRT_LIBRARIES_PACKAGE)
        if package is None:
            return
        directory = Path(package.submodule_search_locations[0])
        for pattern in TENSORRT_LIBRARY_PATTERNS:
            for library in directory.glob(pattern):
                ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)

    def _forward_classification(self, frames):
        """Handle inference for classification models."""
        is_batch = frames.ndim == 4
        img_array = np.array(frames, dtype=np.float32) / 255.0
        if is_batch:
            img_array = np.transpose(
                img_array, (0, 3, 1, 2)
            )  # (B, H, W, C) -> (B, C, H, W)
        else:
            img_array = np.transpose(img_array, (2, 0, 1))  # (H, W, C) -> (C, H, W)
            img_array = np.expand_dims(img_array, 0)
        inputs = {self.input_names[0]: img_array}
        outputs = self.session.run(self.output_names, inputs)
        preds = outputs[0]
        probs = np.exp(preds) / np.sum(np.exp(preds), axis=1, keepdims=True)
        top_classes = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        results = [
            {"labels": [int(c)], "scores": [float(s)]}
            for c, s in zip(top_classes, confidences)
        ]
        return results[0] if not is_batch else results

    def do_forward(self, frames):
        """Handle inference for different types of models, supporting single frames or batches."""
        is_batch = isinstance(frames, np.ndarray) and frames.ndim == 4
        if not isinstance(frames, np.ndarray):
            self.logger.error(f"Invalid input type for forward: {type(frames)}")
            return None

        if self.model_type == "classification":
            return self._forward_classification(frames)

        elif self.model_type == "detection":
            writable_frames = np.array(frames, copy=True, dtype=np.float32) / 255.0
            if is_batch:
                img_array = np.transpose(
                    writable_frames, (0, 3, 1, 2)
                )  # (B, H, W, C) -> (B, C, H, W)
            else:
                img_array = np.transpose(writable_frames, (2, 0, 1))
                img_array = np.expand_dims(img_array, 0)
            inputs = {self.input_names[0]: img_array}
            outputs = self.session.run(self.output_names, inputs)
            # Assuming outputs for detection: [boxes, labels, scores]
            # But based on common format, might be [num_detections, detection_boxes, detection_scores, detection_classes]
            if len(outputs) == 4:
                num_dets, boxes, scores, labels = outputs
                results = []
                for i in range(len(num_dets)):
                    n = int(num_dets[i])
                    res = {
                        "boxes": boxes[i, :n],
                        "labels": labels[i, :n].astype(int),
                        "scores": scores[i, :n],
                    }
                    results.append(res)
            elif len(outputs) == 3:
                boxes, labels, scores = outputs
                results = [
                    {"boxes": boxes, "labels": labels.astype(int), "scores": scores}
                ]
                if is_batch:
                    results = [
                        {
                            "boxes": boxes[j],
                            "labels": labels[j].astype(int),
                            "scores": scores[j],
                        }
                        for j in range(boxes.shape[0])
                    ]
            else:
                self.logger.error("Unexpected output format for detection model.")
                return []
            self.logger.debug(
                f"Batch inference results: {len(results)} frames processed"
            )
            return results[0] if not is_batch else results

        elif self.model_type == "custom":
            # Generic forward for custom ONNX models
            fmt = self.input_format
            if fmt == "auto" and self._input_is_nchw():
                self.input_format = "nchw"
            # Letterbox to the model's fixed input size for inference, keeping
            # the transform so boxes map back to the original frame -- lets the
            # caller feed full-res frames and overlay on them.
            proc, transform = self._letterbox(frames, is_batch)
            img = self._apply_input_format(proc.astype(np.float32) / 255.0, is_batch)
            if "float16" in self.session.get_inputs()[0].type:
                img = img.astype(np.float16)
            outputs = self.session.run(self.output_names, {self.input_names[0]: img})
            raw = outputs if len(outputs) > 1 else outputs[0]
            if isinstance(raw, np.ndarray) and raw.dtype != np.float32:
                raw = raw.astype(np.float32)
            results = self._apply_post_process(raw, is_batch)
            if transform is not None:
                self._unletterbox(results, transform)
            return results

        else:
            raise ValueError("Unsupported model type.")

    def do_generate(self, input_text, max_length=1000, system_prompt=None):
        raise NotImplementedError(
            "ONNX does not support text generation. "
            "Use PyTorch or llama.cpp for LLM workloads."
        )
