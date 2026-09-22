# EngineManager
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


import traceback

from .engine_factory import EngineFactory


class EngineManager:
    def __init__(self, logger, default_engine=EngineFactory.PYTORCH_ENGINE):
        self.logger = logger
        self.engine_name = default_engine
        self.engine = None
        self.device = "cpu"  # Manage device here

    def initialize_engine(self):
        if not self.engine and self.engine_name:
            self.engine = EngineFactory.create(self.engine_name)
            self.engine.do_set_device(self.device)
        if not self.engine:
            self.logger.error(f"Unable to load ML engine: {self.engine_name}")

    def set_device(self, device):
        self.device = device
        if self.engine:
            self.engine.do_set_device(device)

    def do_load_model(self, model_name, **kwargs):
        self.initialize_engine()
        if self.engine is None:
            self.logger.warning(
                f"Cannot load model {model_name}: engine not initialized"
            )
            return False
        if self.engine.model:
            return True
        if model_name is None:
            stack_trace = (
                traceback.format_stack()
            )  # Capture the current call stack as a list of strings
            self.logger.warning(
                f"Cannot load model as model name is not set\n"
                f"Stack trace:\n{''.join(stack_trace)}"  # Join and log the stack trace
            )
            return False
        try:
            self.logger.info(f"Loading model: {model_name}")
            self.engine.do_load_model(model_name, **kwargs)
            self.logger.info(f"Model {model_name} loaded successfully")
            return True
        except Exception:
            self.engine.tokenizer = None
            self.engine.model = None
            raise

    def get_model(self):
        self.initialize_engine()
        if self.engine is None:
            raise ValueError("Engine is not present, unable to get model")
        return self.engine.get_model()

    def get_tokenizer(self):
        if self.engine:
            model = self.get_model()
            if model is None:
                self.logger.warning("Model not loaded yet, attempting to load.")
            return self.engine.tokenizer
        else:
            self.logger.warning("Engine is not present, unable to get the tokenizer.")
            return None
