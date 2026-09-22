# SamEngine
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


class SamEngine(PyTorchEngine):
    """
    PyTorch engine for Segment Anything Model 2 (SAM2).

    Supports HuggingFace model IDs:
      facebook/sam2-hiera-large
      facebook/sam2-hiera-base-plus
      facebook/sam2-hiera-small
      facebook/sam2-hiera-tiny
    """

    def do_load_model(self, model_name, **kwargs):
        try:
            from transformers import Sam2Model, Sam2Processor

            self.processor = Sam2Processor.from_pretrained(model_name)
            self.model = Sam2Model.from_pretrained(model_name)
            self.execute_with_stream(lambda: self.model.to(self.device))
            self.model.eval()
            self.logger.info(f"SAM2 model '{model_name}' loaded on {self.device}")
        except Exception as e:
            raise ValueError(f"Failed to load SAM2 model '{model_name}': {e}")

    def do_forward(self, frames, max_masks=10):
        import numpy as np
        import torch
        from PIL import Image

        is_batch = isinstance(frames, np.ndarray) and frames.ndim == 4
        if not is_batch:
            frames = frames[np.newaxis]

        results = []
        for frame in frames:
            pil_img = Image.fromarray(frame.astype(np.uint8))
            H, W = frame.shape[:2]

            # Automatic mask generation: grid of input points
            grid_size = int(np.ceil(np.sqrt(max_masks)))
            xs = np.linspace(0, W - 1, grid_size).astype(int)
            ys = np.linspace(0, H - 1, grid_size).astype(int)
            points = [[int(x), int(y)] for y in ys for x in xs][:max_masks]
            # sam2 nests points as image, object, point, xy
            input_points = [[[point] for point in points]]

            inputs = self.processor(
                images=pil_img,
                input_points=input_points,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            masks = self.processor.post_process_masks(
                outputs.pred_masks, inputs["original_sizes"]
            )
            scores = outputs.iou_scores

            mask_list = []
            if len(masks) > 0:
                frame_masks = masks[0].cpu().numpy()
                frame_scores = scores[0].cpu().numpy()
                for j in range(min(frame_masks.shape[0], max_masks)):
                    best_idx = frame_scores[j].argmax()
                    mask = frame_masks[j, best_idx]
                    score = float(frame_scores[j, best_idx])
                    mask_list.append(
                        {"mask_idx": j, "score": score, "shape": list(mask.shape)}
                    )

            results.append(
                {
                    "masks": mask_list,
                    "raw_masks": masks[0].cpu().numpy() if len(masks) > 0 else None,
                }
            )
        return results[0] if not is_batch else results
