# EmbeddingEngine
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


# transformers 5 returns a model output here, 4 returned the tensor itself
def projected(features):
    return getattr(features, "pooler_output", features)


class EmbeddingEngine(PyTorchEngine):
    """
    PyTorch engine for image/text embedding extraction.

    Supports CLIP and DINOv2 models via HuggingFace transformers:
      openai/clip-vit-large-patch14  (CLIP — image + text)
      facebook/dinov2-base           (DINOv2 — image only)
    """

    def __init__(self):
        super().__init__()
        self.processor = None
        self.tokenizer = None
        self.output_dim = 0
        self._is_clip = False

    def do_load_model(self, model_name, **kwargs):
        try:

            if "clip" in model_name.lower():
                self._load_clip(model_name)
            elif "dino" in model_name.lower():
                self._load_dinov2(model_name)
            else:
                # Default to CLIP-style loading
                self._load_clip(model_name)

            self.execute_with_stream(lambda: self.model.to(self.device))
            self.model.eval()
            self.logger.info(
                f"Embedding model '{model_name}' loaded on {self.device} "
                f"(dim={self.output_dim})"
            )
        except Exception as e:
            raise ValueError(f"Failed to load embedding model '{model_name}': {e}")

    def _load_clip(self, model_name):
        from transformers import CLIPModel, CLIPProcessor

        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self._is_clip = True
        # Determine output dim from config
        self.output_dim = self.model.config.projection_dim

    def _load_dinov2(self, model_name):
        from transformers import AutoModel, AutoImageProcessor

        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self._is_clip = False
        self.output_dim = self.model.config.hidden_size

    def do_forward(self, frame, normalize=True):
        """
        Extract an embedding vector from a video frame.

        Args:
            frame: numpy RGB array (H, W, 3).
            normalize: if True, L2-normalize the embedding.

        Returns:
            numpy float32 array of shape (output_dim,), or None on failure.
        """
        import numpy as np
        import torch
        from PIL import Image

        try:
            pil_img = Image.fromarray(frame.astype(np.uint8))
            inputs = self.processor(images=pil_img, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                if self._is_clip:
                    emb = projected(self.model.get_image_features(**inputs))
                else:
                    outputs = self.model(**inputs)
                    # Use CLS token embedding
                    emb = outputs.last_hidden_state[:, 0]

            emb = emb.squeeze(0).cpu().numpy().astype(np.float32)
            if normalize:
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
            return emb
        except Exception as e:
            self.logger.error(f"Embedding inference error: {e}")
            return None

    def do_text_embedding(self, text, normalize=True):
        """
        Extract a text embedding (CLIP only).

        Args:
            text: input string.
            normalize: if True, L2-normalize the embedding.

        Returns:
            numpy float32 array of shape (output_dim,), or None.
        """
        import numpy as np
        import torch

        if not self._is_clip:
            self.logger.warning("Text embeddings only supported for CLIP models")
            return None

        try:
            inputs = self.processor(text=[text], return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                emb = projected(self.model.get_text_features(**inputs))
            emb = emb.squeeze(0).cpu().numpy().astype(np.float32)
            if normalize:
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
            return emb
        except Exception as e:
            self.logger.error(f"Text embedding error: {e}")
            return None
