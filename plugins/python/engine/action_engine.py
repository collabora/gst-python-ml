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

from .onnx_export import (
    exported_model_path,
    model_input_frames,
    model_input_shape,
    pixel_normalizer,
)
from .pytorch_engine import PyTorchEngine

TOP_ACTION_COUNT = 5


def action_result(logits, id2label):
    import numpy as np

    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    top_indices = np.argsort(probabilities)[::-1][:TOP_ACTION_COUNT]
    top5 = [
        {
            "label": id2label.get(int(index), f"class_{index}"),
            "score": float(probabilities[index]),
        }
        for index in top_indices
    ]
    return {"label": top5[0]["label"], "score": top5[0]["score"], "top5": top5}


class ExportedAction:
    def __init__(self, model_name, engine_name):
        from transformers import AutoConfig, AutoImageProcessor

        self.engine = None
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.path = exported_model_path(
            engine_name,
            f"{model_name.replace('/', '--')}-{self.config.num_frames}-frames",
            lambda: self._build_graph(model_name),
        )

    def _build_graph(self, model_name):
        import torch
        from transformers import VideoMAEForVideoClassification

        model = VideoMAEForVideoClassification.from_pretrained(model_name)
        normalize = pixel_normalizer(
            self.image_processor.image_mean, self.image_processor.image_std
        )

        # a builtin engine feeds the window of frames as one batch of images
        class ActionGraph(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model

            def forward(self, frames):
                return self.model(pixel_values=normalize(frames).unsqueeze(0)).logits

        frame_count, height, width, channels = model_input_shape(
            self.image_processor, self.config.num_frames
        )
        return ActionGraph(), torch.rand(frame_count, channels, height, width)

    def _model_input(self, frame_buffer):
        return model_input_frames(self.image_processor, list(frame_buffer))

    def do_forward(self, frame_buffer):
        import numpy as np

        logits = self.engine.do_forward(self._model_input(frame_buffer))
        return action_result(np.asarray(logits).reshape(-1), self.config.id2label)


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

        logits = outputs.logits[0].cpu().numpy()
        return action_result(logits, self.model.config.id2label)
