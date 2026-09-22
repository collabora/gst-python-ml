# ActionEngine
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


class ActionEngine(PyTorchEngine):
    """
    PyTorch engine for video action recognition using VideoMAE.

    Supports HuggingFace model IDs:
      MCG-NJU/videomae-base-finetuned-kinetics
      MCG-NJU/videomae-large-finetuned-kinetics
      facebook/timesformer-base-finetuned-k400
    """

    def do_load_model(self, model_name, **kwargs):
        try:
            from transformers import AutoImageProcessor, VideoMAEForVideoClassification

            self.image_processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = VideoMAEForVideoClassification.from_pretrained(model_name)
            self.execute_with_stream(lambda: self.model.to(self.device))
            self.model.eval()
            self.logger.info(f"VideoMAE model '{model_name}' loaded on {self.device}")
        except Exception as e:
            raise ValueError(f"Failed to load VideoMAE model '{model_name}': {e}")

    def do_forward(self, frame_buffer):
        """
        Classify a buffer of frames.

        Args:
            frame_buffer: list of numpy arrays (H, W, 3), length = num_frames

        Returns:
            dict with 'label', 'score', and 'top5' predictions
        """
        import numpy as np
        import torch
        from PIL import Image

        pil_frames = [Image.fromarray(f.astype(np.uint8)) for f in frame_buffer]

        inputs = self.image_processor(pil_frames, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1)
        top5_indices = probs.topk(5).indices.cpu().numpy()
        top5_scores = probs.topk(5).values.cpu().numpy()

        top1_idx = top5_indices[0]
        label = self.model.config.id2label.get(int(top1_idx), f"class_{top1_idx}")
        score = float(top5_scores[0])

        top5 = []
        for idx, s in zip(top5_indices, top5_scores):
            name = self.model.config.id2label.get(int(idx), f"class_{idx}")
            top5.append({"label": name, "score": float(s)})

        return {"label": label, "score": score, "top5": top5}
