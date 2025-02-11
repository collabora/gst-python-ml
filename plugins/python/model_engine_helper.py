# ModelEngineHelper
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

from abc import abstractmethod
from engine.engine_factory import EngineFactory
from log.logger_factory import LoggerFactory


class ModelEngineHelper:
    def __init__(self, logger, default_engine=EngineFactory.PYTORCH_ENGINE):
        self.logger = logger
        self.engine_name = default_engine
        self.engine = None
        self.kwargs = {}
        self.device = "cpu"  # Manage device here

    def initialize_engine(self, engine_name):
        if engine_name:
            self.engine = EngineFactory.create_engine(engine_name, self.device)
        else:
            self.logger.error(f"Unsupported ML engine: {engine_name}")

    def set_device(self, device):
        self.device = device
        if self.engine:
            self.engine.set_device(device)

    def load_model(self, model_name):
        if self.engine and model_name:
            self.engine.load_model(model_name, **self.kwargs)
        else:
            self.logger.warning(
                "Engine is not present or model name not provided, unable to load the model."
            )

    def get_model(self):
        if self.engine:
            return self.engine.get_model()
        else:
            self.logger.warning("Engine is not present, unable to get the model.")
            return None

    def set_model(self, model):
        if self.engine:
            self.engine.set_model(model)
        else:
            self.logger.warning("Engine is not present, unable to set the model.")

    def get_tokenizer(self):
        if self.engine:
            model = self.get_model()
            if model is None:
                self.logger.warning("Model not loaded yet, attempting to load.")
            return self.engine.tokenizer
        else:
            self.logger.warning("Engine is not present, unable to get the tokenizer.")
            return None

    def update_engine_properties(self, batch_size, frame_stride, device_queue_id):
        if self.engine:
            self.engine.batch_size = batch_size
            self.engine.frame_stride = frame_stride
            self.engine.device_queue_id = device_queue_id
