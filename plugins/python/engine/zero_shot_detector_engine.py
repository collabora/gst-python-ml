# ZeroShotDetectorEngine
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

DEFAULT_CONFIDENCE = 0.1


class ZeroShotDetectorEngine(PyTorchEngine):
    def __init__(self):
        super().__init__()
        self.labels = []
        self.confidence = DEFAULT_CONFIDENCE

    def do_load_model(self, model_name, **kwargs):
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

            self.image_processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
            self.execute_with_stream(lambda: self.model.to(self.device))
            self.model.eval()
            self.logger.info(
                f"Zero-shot detection model '{model_name}' loaded on {self.device}"
            )
        except Exception as e:
            raise ValueError(
                f"Failed to load zero-shot detection model '{model_name}': {e}"
            )

    def do_forward(self, frames):
        import numpy as np
        import torch
        from PIL import Image

        if not self.labels:
            raise ValueError("no labels set for zero-shot detection")
        if self.model is None or self.image_processor is None:
            raise ValueError("zero-shot detection model is not loaded")

        is_batch = isinstance(frames, np.ndarray) and frames.ndim == 4
        batch = frames if is_batch else np.expand_dims(frames, 0)
        images = [Image.fromarray(frame.astype(np.uint8)) for frame in batch]
        candidate_labels = [self.labels] * len(images)

        inputs = self.image_processor(
            images=images,
            text=candidate_labels,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.execute_with_stream(lambda: self.model(**inputs))

        processed = self.image_processor.post_process_grounded_object_detection(
            outputs=outputs,
            threshold=self.confidence,
            target_sizes=[(image.height, image.width) for image in images],
            text_labels=candidate_labels,
        )
        results = [self._detections(item) for item in processed]
        self.logger.debug(
            f"Zero-shot detections per frame: {[len(r['boxes']) for r in results]}"
        )
        return results if is_batch else results[0]

    def _detections(self, processed):
        boxes = [[float(value) for value in box] for box in processed["boxes"]]
        scores = [float(score) for score in processed["scores"]]
        indices = self._label_indices(processed)
        kept = [i for i, index in enumerate(indices) if index is not None]
        return {
            "boxes": [boxes[i] for i in kept],
            "labels": [indices[i] for i in kept],
            "scores": [scores[i] for i in kept],
        }

    def _label_indices(self, processed):
        text_labels = processed.get("text_labels")
        if text_labels is None:
            return [int(label) for label in processed["labels"]]
        return [self._index_of(text) for text in text_labels]

    def _index_of(self, text):
        # grounding dino reports the phrase it decoded, not an index into the labels
        phrase = text.strip().lower()
        if not phrase:
            return None
        for index, label in enumerate(self.labels):
            if label.strip().lower() == phrase:
                return index
        for index, label in enumerate(self.labels):
            if phrase in label.strip().lower():
                return index
        return None
