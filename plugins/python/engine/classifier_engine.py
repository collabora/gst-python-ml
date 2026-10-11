# Classifier
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

from .onnx_export import exported_model_path, pixel_normalizer
from .pytorch_engine import PyTorchEngine

COLOR_CHANNELS = 3
BATCH_DIMENSIONS = 4


def default_weights(model_name):
    from torchvision import models

    return models.get_model_weights(model_name).DEFAULT


def classification(categories, index, score):
    return {"label": categories[index], "score": float(score)}


def top_class(logits, categories):
    import numpy as np

    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    index = int(probabilities.argmax())
    return classification(categories, index, probabilities[index])


class ExportedClassifier:
    def __init__(self, model_name, engine_name):
        self.engine = None
        weights = default_weights(model_name)
        self.transforms = weights.transforms()
        self.categories = weights.meta["categories"]
        self.path = exported_model_path(
            engine_name,
            f"{model_name}-classifier",
            lambda: self._build_graph(model_name, weights),
        )

    def _build_graph(self, model_name, weights):
        import torch
        from torchvision import models

        model = models.get_model(model_name, weights=weights)
        normalize = pixel_normalizer(self.transforms.mean, self.transforms.std)

        class ClassifierGraph(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model

            def forward(self, image):
                return self.model(normalize(image))

        crop_size = self.transforms.crop_size[0]
        example_input = torch.rand(1, COLOR_CHANNELS, crop_size, crop_size)
        return ClassifierGraph(), example_input

    def do_forward(self, frames):

        if frames.ndim == BATCH_DIMENSIONS:
            return [self._classify(frame) for frame in frames]
        return self._classify(frames)

    # weights.transforms() without the normalize the graph runs
    def _classify(self, frame):
        import numpy as np
        from PIL import Image
        from torchvision.transforms import functional

        image = functional.resize(
            Image.fromarray(frame),
            self.transforms.resize_size,
            interpolation=self.transforms.interpolation,
            antialias=self.transforms.antialias,
        )
        crop = np.asarray(functional.center_crop(image, self.transforms.crop_size))
        logits = np.asarray(self.engine.do_forward(crop), dtype=np.float32).ravel()
        return top_class(logits, self.categories)


class ClassifierEngine(PyTorchEngine):
    def do_load_model(self, model_name, **kwargs):
        from torchvision import models

        try:
            weights = default_weights(model_name)
            self.model = models.get_model(model_name, weights=weights)
        except Exception as error:
            raise ValueError(f"Failed to load classifier '{model_name}': {error}")
        self.model.eval()
        self.model.to(self.device)
        self.transforms = weights.transforms()
        self.categories = weights.meta["categories"]
        self.logger.info(f"Classifier '{model_name}' loaded on {self.device}")

    def do_forward(self, frames):
        if frames.ndim == BATCH_DIMENSIONS:
            return [self._classify(frame) for frame in frames]
        return self._classify(frames)

    def _classify(self, frame):
        import torch
        from PIL import Image

        image = (
            frame.permute(2, 0, 1) if torch.is_tensor(frame) else Image.fromarray(frame)
        )
        model_input = self.transforms(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(model_input)[0]
        return top_class(logits.cpu().numpy(), self.categories)
