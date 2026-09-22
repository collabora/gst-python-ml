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

from .pytorch_engine import PyTorchEngine


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

                # Global average pool to get a feature vector
                feat_vec = features.mean(dim=[2, 3]).squeeze(0).cpu().numpy()

                # Compute anomaly score against reference distribution
                anomaly_score = 0.0
                if self.reference_features is not None:
                    distances = np.linalg.norm(
                        self.reference_features - feat_vec, axis=-1
                    )
                    anomaly_score = float(distances.min())

                # Generate a spatial anomaly heatmap from feature map distances
                feat_map = features.squeeze(0).cpu().numpy()
                heatmap = np.linalg.norm(feat_map, axis=0)
                heatmap = (heatmap - heatmap.min()) / (
                    heatmap.max() - heatmap.min() + 1e-8
                )

                is_anomaly = anomaly_score >= threshold

                results.append(
                    {
                        "score": anomaly_score,
                        "is_anomaly": is_anomaly,
                        "heatmap": heatmap,
                    }
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
