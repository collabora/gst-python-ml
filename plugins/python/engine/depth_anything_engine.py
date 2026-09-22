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

from .pytorch_engine import PyTorchEngine


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
