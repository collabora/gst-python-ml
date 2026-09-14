# MLEngine
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

from abc import ABC, abstractmethod

from log.logger_factory import LoggerFactory


class MLEngine(ABC):
    """Abstract base class for machine learning engines that load models, run inference on image frames,
    and generate text with language models."""

    def __init__(self):
        self.logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)
        self.device = None
        self.device_index = 0
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.batch_size = 1  # Default batch size
        self.frame_buffer = []  # For vision-text models or manual buffering
        self.frame_stride = None
        self.counter = 0
        self.device_queue_id = None
        self.track = False
        self.prompt = "What is shown in this image?"  # Default prompt
        self.input_format = "auto"  # "auto", "nhwc", "nchw"
        self.post_process = (
            "auto"  # "auto", "none", or a format key from detection_decoder
        )

    # Interface #
    @abstractmethod
    def do_load_model(self, model_name, **kwargs):
        """Load a model by name or path, with additional options."""
        pass

    @abstractmethod
    def do_set_device(self, device):
        """Set the device (e.g., cpu, cuda)."""
        pass

    @abstractmethod
    def do_forward(self, frames):
        """Execute inference on a single frame or batch of frames.
        Input can be a single NumPy array (H, W, C) or a batch (B, H, W, C)."""
        pass

    @abstractmethod
    def do_generate(self, input_text, max_length=1000, system_prompt=None):
        """Generate LLM text."""
        pass

    # Implementation #
    def _apply_input_format(self, img, is_batch):
        """Normalize input to (B, ?, H, W) or (B, H, W, C) per self.input_format."""
        import numpy as np

        if not is_batch:
            img = np.expand_dims(img, axis=0)
        if self.input_format == "nchw":
            img = np.transpose(img, (0, 3, 1, 2))
        # "nhwc" or "auto" → leave as (B, H, W, C)
        return img

    def _apply_post_process(self, raw, is_batch):
        """Apply post-processing to raw engine output per self.post_process."""
        import numpy as np

        pp = self.post_process
        if pp == "auto":
            if (
                isinstance(raw, np.ndarray)
                and raw.ndim == 3
                and raw.shape[1] >= 5
                and raw.shape[2] > raw.shape[1]
            ):
                pp = "anchor_free"
            else:
                pp = "none"
        if pp != "none" and not isinstance(raw, list):
            from utils.detection_decoder import decode

            results = decode(
                raw,
                pp,
                conf_threshold=getattr(self, "conf", 0.25),
                iou_threshold=getattr(self, "iou", 0.45),
            )
            return results[0] if not is_batch else results
        return raw

    def set_prompt(self, prompt):
        """Set the custom prompt for generating responses."""
        self.prompt = prompt

    def get_prompt(self):
        """Return the custom prompt."""
        return self.prompt

    def get_device(self):
        """Return the device the model is running on."""
        return self.device

    def get_model(self):
        """Return the loaded model for use in inference."""
        return self.model
