# FaceEngine
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

import os

from .pytorch_engine import PyTorchEngine


class FaceEngine(PyTorchEngine):
    """
    PyTorch engine for face detection and recognition using InsightFace.

    Supports InsightFace model packs:
      buffalo_l   (large, most accurate)
      buffalo_s   (small, fastest)
      buffalo_sc  (small with recognition)
    """

    def do_load_model(self, model_name, **kwargs):
        try:
            from insightface.app import FaceAnalysis

            self.app = FaceAnalysis(
                name=model_name,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            self.gallery = {}
            self.logger.info(f"InsightFace model '{model_name}' loaded")
        except Exception as e:
            raise ValueError(f"Failed to load InsightFace model '{model_name}': {e}")

    def load_gallery(self, gallery_path):
        """Load known face embeddings from a directory of images."""
        import numpy as np
        from PIL import Image

        if not gallery_path or not os.path.isdir(gallery_path):
            return

        self.gallery = {}
        for fname in os.listdir(gallery_path):
            fpath = os.path.join(gallery_path, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                img = np.array(Image.open(fpath).convert("RGB"))
                faces = self.app.get(img)
                if faces:
                    name = os.path.splitext(fname)[0]
                    self.gallery[name] = faces[0].embedding
                    self.logger.info(f"Loaded gallery face: {name}")
            except Exception as e:
                self.logger.warning(f"Failed to load gallery image '{fname}': {e}")

    def do_forward(self, frames, threshold=0.5):
        import numpy as np

        is_batch = isinstance(frames, np.ndarray) and frames.ndim == 4
        if not is_batch:
            frames = frames[np.newaxis]

        results = []
        for frame in frames:
            faces = self.app.get(frame.astype(np.uint8))
            detections = []
            for face in faces:
                bbox = face.bbox.astype(float).tolist()
                score = float(face.det_score)
                embedding = face.embedding

                identity = "unknown"
                best_sim = 0.0
                if self.gallery and embedding is not None:
                    for name, gallery_emb in self.gallery.items():
                        sim = float(
                            np.dot(embedding, gallery_emb)
                            / (
                                np.linalg.norm(embedding) * np.linalg.norm(gallery_emb)
                                + 1e-8
                            )
                        )
                        if sim > best_sim:
                            best_sim = sim
                            if sim >= threshold:
                                identity = name

                detections.append(
                    {
                        "bbox": bbox,
                        "score": score,
                        "identity": identity,
                        "similarity": best_sim,
                    }
                )
            results.append(detections)
        return results[0] if not is_batch else results
