# ClipEngine
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


class ClipEngine(PyTorchEngine):
    """
    PyTorch engine for CLIP and SigLIP zero-shot image classification.

    Works with any HuggingFace CLIP-compatible model:
      openai/clip-vit-base-patch32
      openai/clip-vit-large-patch14
      google/siglip-base-patch16-224
      google/siglip-large-patch16-384
    """

    def __init__(self):
        super().__init__()
        self._labels = []

    @property
    def clip_labels(self):
        return self._labels

    @clip_labels.setter
    def clip_labels(self, value):
        self._labels = value

    def do_load_model(self, model_name, **kwargs):
        try:
            from transformers import AutoProcessor, AutoModel

            self.image_processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.execute_with_stream(lambda: self.model.to(self.device))
            self.model.eval()
            self.logger.info(f"CLIP model '{model_name}' loaded on {self.device}")
        except Exception as e:
            raise ValueError(f"Failed to load CLIP model '{model_name}': {e}")

    def do_forward(self, frame):
        """
        Run zero-shot classification.

        Args:
            frame: RGB numpy array [H, W, 3]

        Returns:
            List of (label, probability) tuples sorted by probability descending,
            or None if no labels are set.
        """
        import numpy as np
        import torch
        from PIL import Image

        if not self._labels:
            self.logger.warning("No labels set — set the 'labels' property")
            return None

        pil_img = Image.fromarray(frame.astype(np.uint8))
        inputs = self.image_processor(
            text=self._labels,
            images=pil_img,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # logits_per_image: [1, num_labels]
        probs = outputs.logits_per_image.softmax(dim=1)[0]
        results = [(label, prob.item()) for label, prob in zip(self._labels, probs)]
        results.sort(key=lambda x: x[1], reverse=True)
        return results
