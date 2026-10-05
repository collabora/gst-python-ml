# AnomalyEngine
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

from .ml_engine import TORCHVISION_WEIGHTS
from .onnx_export import cached_onnx_export, pixel_normalizer
from .pytorch_engine import PyTorchEngine

COLOR_CHANNELS = 3
BACKBONE_INPUT_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# the layers after these pool the feature map away
POOLING_LAYER_COUNT = 2
HEATMAP_EPSILON = 1e-8


def anomaly_result(feature_map, reference_features, threshold):
    import numpy as np

    feature_vector = feature_map.mean(axis=(1, 2))
    anomaly_score = 0.0
    if reference_features is not None:
        distances = np.linalg.norm(reference_features - feature_vector, axis=-1)
        anomaly_score = float(distances.min())
    heatmap = np.linalg.norm(feature_map, axis=0)
    heatmap = (heatmap - heatmap.min()) / (
        heatmap.max() - heatmap.min() + HEATMAP_EPSILON
    )
    return {
        "score": anomaly_score,
        "is_anomaly": anomaly_score >= threshold,
        "heatmap": heatmap,
    }


class ExportedAnomaly:
    def __init__(self, model_name):
        self.engine = None
        self.reference_features = None
        self.path = cached_onnx_export(
            f"{model_name}-anomaly-features", lambda: self._build_graph(model_name)
        )

    def _build_graph(self, model_name):
        import torch
        import torchvision.models as models

        backbone = models.get_model(model_name, weights=TORCHVISION_WEIGHTS)
        feature_layers = torch.nn.Sequential(
            *list(backbone.children())[:-POOLING_LAYER_COUNT]
        )
        normalize = pixel_normalizer(IMAGENET_MEAN, IMAGENET_STD)

        class AnomalyFeatures(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.feature_layers = feature_layers

            def forward(self, image):
                return self.feature_layers(normalize(image))

        example_input = torch.rand(
            1, COLOR_CHANNELS, BACKBONE_INPUT_SIZE, BACKBONE_INPUT_SIZE
        )
        return AnomalyFeatures(), example_input

    def load_reference(self, reference_path):
        import numpy as np

        self.reference_features = np.load(reference_path)

    def do_forward(self, frame, threshold=0.5):
        import numpy as np
        from PIL import Image

        model_size = (BACKBONE_INPUT_SIZE, BACKBONE_INPUT_SIZE)
        resized = Image.fromarray(frame).resize(model_size, Image.BILINEAR)
        model_input = np.asarray(resized)
        feature_map = np.asarray(self.engine.do_forward(model_input))[0]
        return anomaly_result(feature_map, self.reference_features, threshold)


class AnomalyEngine(PyTorchEngine):
    """
    PyTorch engine for anomaly detection using a PatchCore-like approach.

    Uses a pretrained feature extractor (WideResNet50 or ResNet) to extract
    patch-level features and compare them against a reference distribution.

    Supports torchvision backbone models:
      wide_resnet50_2
      resnet50
      resnet18
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_layers = None
        self.reference_features = None
        self._transform = None

    def do_load_model(self, model_name, **kwargs):
        try:
            import torch
            import torchvision.models as models

            model_fn = getattr(models, model_name, None)
            if model_fn is None:
                raise ValueError(f"Unknown backbone model: {model_name}")

            self.backbone = model_fn(weights="DEFAULT")
            # Remove the final FC layer to get feature maps
            self.feature_layers = torch.nn.Sequential(
                *list(self.backbone.children())[:-2]
            )
            self.execute_with_stream(lambda: self.feature_layers.to(self.device))
            self.feature_layers.eval()

            self.reference_features = None
            self._transform = None
            self.logger.info(f"Anomaly backbone '{model_name}' loaded on {self.device}")
        except Exception as e:
            raise ValueError(f"Failed to load anomaly backbone '{model_name}': {e}")

    def load_reference(self, reference_path):
        """Load precomputed reference features from a .npy file."""
        import numpy as np

        try:
            self.reference_features = np.load(reference_path)
            self.logger.info(
                f"Loaded reference features from '{reference_path}': "
                f"shape={self.reference_features.shape}"
            )
        except Exception as e:
            self.logger.warning(f"Failed to load reference features: {e}")

    def _get_transform(self):
        if self.feature_layers is None:
            raise ValueError("anomaly backbone is not loaded")
        if self._transform is None:
            from torchvision import transforms

            self._transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        return self._transform

    def do_forward(self, frames, threshold=0.5):
        import numpy as np
        import torch

        is_batch = isinstance(frames, np.ndarray) and frames.ndim == 4
        if not is_batch:
            frames = frames[np.newaxis]

        transform = self._get_transform()
        results = []
        for frame in frames:
            try:
                tensor = transform(frame.astype(np.uint8)).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    features = self.feature_layers(tensor)

                feature_map = features.squeeze(0).cpu().numpy()
                results.append(
                    anomaly_result(feature_map, self.reference_features, threshold)
                )
            except Exception as e:
                self.logger.error(f"Anomaly inference error on frame: {e}")
                results.append(
                    {
                        "score": 0.0,
                        "is_anomaly": False,
                        "heatmap": None,
                    }
                )

        return results[0] if not is_batch else results
