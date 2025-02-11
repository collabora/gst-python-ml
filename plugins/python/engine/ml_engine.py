# MLEngine
# Copyright (C) 2024-2025 Collabora Ltd.
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
    def __init__(self, device="cpu"):
        self.logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)
        self.device = device
        self.device_index = 0
        self.model = None
        self.vision_language_model = False
        self.tokenizer = None
        self.image_processor = None
        self.batch_size = 1
        self.frame_buffer = []
        self.frame_stride = None
        self.counter = 0
        self.device_queue_id = None
        self.track = False
        self.prompt = "What is shown in this image?"  # Default prompt

    @abstractmethod
    def load_model(self, model_name, **kwargs):
        """Load a model by name or path, with additional options."""
        pass

    def set_prompt(self, prompt):
        """Set the custom prompt for generating responses."""
        self.prompt = prompt

    def get_prompt(self):
        """Return the custom prompt."""
        return self.prompt

    def get_device(self):
        """Return the loaded model for use in inference."""
        return self.device

    @abstractmethod
    def set_device(self, device):
        """Set the device (e.g., cpu, cuda)."""
        pass

    def get_model(self):
        """Return the loaded model for use in inference."""
        return self.model

    def set_model(self, model):
        """Set the model directly (useful for loading pre-built models)."""
        self.model = model

    @abstractmethod
    def forward(self, frame):
        """Execute inference (usually object detection)"""
        pass

    @abstractmethod
    def generate(self, input_text, max_length=100):
        """Generate LLM text"""
        pass
