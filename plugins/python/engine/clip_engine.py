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

from .jax_engine import keras_hub_preset
from .onnx_export import (
    cached_onnx_export,
    model_input_frames,
    model_input_shape,
    pixel_normalizer,
)
from .pytorch_engine import PyTorchEngine, projected


class ExportedClip:
    def __init__(self, model_name):
        from transformers import AutoModel, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval()
        self.logit_scale = float(self.model.logit_scale.exp())
        self.text_embeddings_by_labels = {}
        self.clip_labels = []
        self.engine = None
        self.path = self._model_path(model_name)

    def _model_path(self, model_name):
        return cached_onnx_export(
            f"{model_name.replace('/', '--')}-image-encoder", self._build_image_encoder
        )

    def _build_image_encoder(self):
        import torch

        model = self.model
        image_processor = self.processor.image_processor
        normalize = pixel_normalizer(
            image_processor.image_mean, image_processor.image_std
        )

        class ClipImageEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model

            def forward(self, image):
                features = projected(
                    self.model.get_image_features(pixel_values=normalize(image))
                )
                return torch.nn.functional.normalize(features, dim=-1)

        height, width, channels = model_input_shape(image_processor)
        return ClipImageEncoder(), torch.rand(1, channels, height, width)

    def _model_input(self, frame):
        return model_input_frames(self.processor.image_processor, frame)

    def _text_embeddings(self, labels):
        import torch

        key = tuple(labels)
        if key not in self.text_embeddings_by_labels:
            tokens = self.processor(text=labels, return_tensors="pt", padding=True)
            with torch.no_grad():
                features = projected(self.model.get_text_features(**tokens))
            embeddings = torch.nn.functional.normalize(features, dim=-1)
            self.text_embeddings_by_labels[key] = embeddings.numpy()
        return self.text_embeddings_by_labels[key]

    def do_forward(self, frame):
        import numpy as np

        labels = self.clip_labels
        if not labels:
            return None
        embedding = self.engine.do_forward(self._model_input(frame))
        image_embedding = np.asarray(embedding).reshape(-1)
        image_embedding = image_embedding / np.linalg.norm(image_embedding)
        logits = self.logit_scale * self._text_embeddings(labels) @ image_embedding
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        results = list(zip(labels, probabilities.tolist()))
        results.sort(key=lambda result: result[1], reverse=True)
        return results


# keras-hub's clip tokenizer needs tensorflow
class KerasHubClip(ExportedClip):
    def _model_path(self, model_name):
        return keras_hub_preset(model_name)


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
