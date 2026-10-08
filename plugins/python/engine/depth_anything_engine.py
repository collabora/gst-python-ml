# DepthAnythingEngine
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

from .jax_engine import keras_hub_preset
from .onnx_export import exported_model_path, pixel_normalizer
from .pytorch_engine import PyTorchEngine

COLOR_CHANNELS = 3
# the size the model was trained at
EXPORTED_INPUT_SIZE = 518
BATCH_DIMENSIONS = 4


class ExportedDepthAnything:
    def __init__(self, model_name, engine_name):
        self.engine = None
        self.path = self._model_path(model_name, engine_name)

    def _model_path(self, model_name, engine_name):
        return exported_model_path(
            engine_name,
            f"{model_name.replace('/', '--')}-{EXPORTED_INPUT_SIZE}",
            lambda: self._build_graph(model_name),
        )

    def _build_graph(self, model_name):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForDepthEstimation.from_pretrained(model_name)
        normalize = pixel_normalizer(processor.image_mean, processor.image_std)

        class DepthAnythingGraph(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model

            def forward(self, image):
                return self.model(pixel_values=normalize(image)).predicted_depth

        example_input = torch.rand(
            1, COLOR_CHANNELS, EXPORTED_INPUT_SIZE, EXPORTED_INPUT_SIZE
        )
        return DepthAnythingGraph(), example_input

    def do_forward(self, frames):
        if frames.ndim == BATCH_DIMENSIONS:
            return [self._depth_map(frame) for frame in frames]
        return self._depth_map(frames)

    def _depth_map(self, frame):
        import cv2
        import numpy as np

        height, width = frame.shape[:2]
        model_size = (EXPORTED_INPUT_SIZE, EXPORTED_INPUT_SIZE)
        model_input = cv2.resize(frame, model_size, interpolation=cv2.INTER_CUBIC)
        depth_map = np.asarray(self.engine.do_forward(model_input), dtype=np.float32)
        depth_map = depth_map.reshape(model_size)
        return cv2.resize(depth_map, (width, height), interpolation=cv2.INTER_CUBIC)


class KerasHubDepthAnything(ExportedDepthAnything):
    def _model_path(self, model_name, engine_name):
        return keras_hub_preset(model_name)


class DepthAnythingEngine(PyTorchEngine):
    """
    PyTorch engine for DepthAnything V2 monocular depth estimation.

    Supports HuggingFace model IDs:
      depth-anything/Depth-Anything-V2-Small-hf  (fastest)
      depth-anything/Depth-Anything-V2-Base-hf
      depth-anything/Depth-Anything-V2-Large-hf  (most accurate)
    """

    def do_load_model(self, model_name, **kwargs):
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            self.image_processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
            self.execute_with_stream(lambda: self.model.to(self.device))
            self.model.eval()
            self.logger.info(
                f"DepthAnything model '{model_name}' loaded on {self.device}"
            )
        except Exception as e:
            raise ValueError(f"Failed to load depth model '{model_name}': {e}")

    def do_forward(self, frames):
        import numpy as np
        import torch
        import torch.nn.functional as F
        from PIL import Image

        is_batch = isinstance(frames, np.ndarray) and frames.ndim == 4
        if not is_batch:
            frames = frames[np.newaxis]

        results = []
        for frame in frames:
            pil_img = Image.fromarray(frame.astype(np.uint8))
            H, W = frame.shape[:2]
            inputs = self.image_processor(images=pil_img, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            # outputs.predicted_depth: [1, H', W']
            depth_up = F.interpolate(
                outputs.predicted_depth.unsqueeze(0),
                size=(H, W),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            results.append(depth_up.cpu().numpy())
        return results[0] if not is_batch else results
