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

from .onnx_export import (
    exported_model_path,
    model_input_frames,
    model_input_shape,
    pixel_normalizer,
)
from .pytorch_engine import PyTorchEngine

DEFAULT_MAX_MASKS = 10
BATCH_DIMENSIONS = 4
DECODER_DEVICE = "cpu"


# sam2 nests points as image, object, point, xy
def grid_input_points(height, width, max_masks):
    import numpy as np

    grid_size = int(np.ceil(np.sqrt(max_masks)))
    xs = np.linspace(0, width - 1, grid_size).astype(int)
    ys = np.linspace(0, height - 1, grid_size).astype(int)
    points = [[int(x), int(y)] for y in ys for x in xs][:max_masks]
    return [[[point] for point in points]]


def segmentation_result(masks, scores, max_masks):
    mask_list = []
    if len(masks) > 0:
        frame_masks = masks[0].cpu().numpy()
        frame_scores = scores[0].cpu().numpy()
        for mask_index in range(min(frame_masks.shape[0], max_masks)):
            best_index = frame_scores[mask_index].argmax()
            mask = frame_masks[mask_index, best_index]
            score = float(frame_scores[mask_index, best_index])
            mask_list.append(
                {"mask_idx": mask_index, "score": score, "shape": list(mask.shape)}
            )
    return {
        "masks": mask_list,
        "raw_masks": masks[0].cpu().numpy() if len(masks) > 0 else None,
    }


class ExportedSam:
    def __init__(self, model_name, engine_name):
        self.engine = None
        # the prompt encoder and mask decoder stay on pytorch
        self.torch_engine = SamEngine()
        self.torch_engine.do_set_device(DECODER_DEVICE)
        self.torch_engine.do_load_model(model_name)
        self.path = exported_model_path(
            engine_name,
            f"{model_name.replace('/', '--')}-image-encoder",
            self._build_image_encoder,
        )

    @property
    def _image_processor(self):
        return self.torch_engine.processor.image_processor

    def _build_image_encoder(self):
        import torch

        model = self.torch_engine.model
        normalize = pixel_normalizer(
            self._image_processor.image_mean, self._image_processor.image_std
        )

        class Sam2ImageEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model

            def forward(self, image):
                return tuple(self.model.get_image_embeddings(normalize(image)))

        height, width, channels = model_input_shape(self._image_processor)
        return Sam2ImageEncoder(), torch.rand(1, channels, height, width)

    def _segment(self, frame, max_masks):
        import torch

        processor = self.torch_engine.processor
        height, width = frame.shape[:2]
        image_embeddings = self.engine.do_forward(
            model_input_frames(self._image_processor, frame)
        )
        prompts = processor(
            original_sizes=[[height, width]],
            input_points=grid_input_points(height, width, max_masks),
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.torch_engine.model(
                image_embeddings=[
                    torch.from_numpy(embedding) for embedding in image_embeddings
                ],
                input_points=prompts["input_points"],
            )
        masks = processor.post_process_masks(
            outputs.pred_masks, prompts["original_sizes"]
        )
        return segmentation_result(masks, outputs.iou_scores, max_masks)

    def do_forward(self, frames, max_masks=DEFAULT_MAX_MASKS):
        if frames.ndim == BATCH_DIMENSIONS:
            return [self._segment(frame, max_masks) for frame in frames]
        return self._segment(frames, max_masks)


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

    def do_forward(self, frames, max_masks=DEFAULT_MAX_MASKS):
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

            inputs = self.processor(
                images=pil_img,
                input_points=grid_input_points(H, W, max_masks),
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            masks = self.processor.post_process_masks(
                outputs.pred_masks, inputs["original_sizes"]
            )
            results.append(segmentation_result(masks, outputs.iou_scores, max_masks))
        return results[0] if not is_batch else results
