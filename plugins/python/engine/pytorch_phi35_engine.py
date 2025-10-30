# PyTorchPhi35Engine
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


import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
)
import numpy as np
from .pytorch_engine import PyTorchEngine


class PyTorchPhi35Engine(PyTorchEngine):
    def load_model(self, model_name, **kwargs):
        """Load a Phi-3-vision model from Hugging Face."""
        processor_name = kwargs.get("processor_name")
        tokenizer_name = kwargs.get("tokenizer_name")

        try:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Phi-3.5-vision-instruct",
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                _attn_implementation="flash_attention_2",
            )
            self.processor = AutoProcessor.from_pretrained(
                "microsoft/Phi-3.5-vision-instruct", trust_remote_code=True
            )
            self.logger.info("Phi-3.5-vision model and processor loaded successfully.")
            self.vision_language_model = True
            self.model.eval()

            # FIXED: Skip .to() for 4-bit models
            if not (
                hasattr(self.model, "is_loaded_in_4bit")
                and self.model.is_loaded_in_4bit
            ):
                self.execute_with_stream(lambda: self.model.to(self.device))
                self.logger.info(f"Model moved to {self.device}")

            return True

        except Exception as e:
            self.logger.error(f"Error loading model '{model_name}': {e}")
            self.tokenizer = None
            self.model = None
            return False
