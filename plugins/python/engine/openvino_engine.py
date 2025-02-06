# OpenVinoEngine
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

import numpy as np
from openvino.runtime import Core
from .ml_engine import MLEngine


class OpenVinoEngine(MLEngine):
    def __init__(self, device="CPU"):
        self.device = device
        self.core = Core()
        self.compiled_model = None
        self.tokenizer = None
        self.is_vision_model = False
        self.is_llm = False

    def load_model(self, model_name, **kwargs):
        """
        Load the model using OpenVINO and differentiate between vision and LLM models.
        """
        try:
            # Load model
            model_path = f"{model_name}.xml"
            self.model = self.core.read_model(model=model_path)
            self.compiled_model = self.core.compile_model(self.model, self.device)
            self.logger.info(
                f"Model '{model_name}' loaded successfully on {self.device}"
            )

            # Inspect input shape to determine the type of model
            input_shape = self.compiled_model.input(0).shape

            if (
                len(input_shape) == 4 and input_shape[1] == 3
            ):  # Expecting image input (batch, 3 channels, height, width)
                self.is_vision_model = True
                self.logger.info("Model identified as a vision model.")
            elif len(input_shape) == 2:  # Expecting LLM input (batch, sequence_length)
                self.is_llm = True
                self.logger.info("Model identified as a large language model (LLM).")

            if self.is_llm:
                # Load a tokenizer for the LLM (Hugging Face for instance)
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        except Exception as e:
            raise ValueError(f"Failed to load model '{model_name}'. Error: {e}")

    def generate(self, input_data, max_length=100):
        """
        Process input for the respective model type.
        """
        if self.is_vision_model:
            # Preprocess input for vision models (e.g., resize, normalize)
            return self.preprocess_vision_input(input_data)
        elif self.is_llm:
            # Preprocess input for LLMs (tokenization)
            return self.preprocess_llm_input(input_data, max_length)
        else:
            raise ValueError("Unknown model type. Please load a model first.")

    def preprocess_vision_input(self, image):
        """
        Preprocess input image for vision models.
        """
        # Example preprocessing: Resize and normalize the image
        input_shape = self.compiled_model.input(0).shape
        resized_image = np.resize(
            image, input_shape[2:]
        )  # Resize image to (height, width)
        resized_image = resized_image.astype(np.float32) / 255.0  # Normalize
        resized_image = np.expand_dims(resized_image, axis=0)  # Add batch dimension
        return resized_image

    def preprocess_llm_input(self, input_text, max_length):
        """
        Tokenize input text for LLM models.
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized. Load an LLM model first.")

        tokens = self.tokenizer(
            input_text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="np",  # Return as NumPy array
        )
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        return input_ids, attention_mask

    def forward(self, input_data):
        """
        Perform inference using the loaded OpenVINO model.
        """
        if self.compiled_model is None:
            raise RuntimeError("Model not loaded. Please load the model first.")

        if self.is_vision_model:
            # Perform inference for vision models
            return self.perform_vision_inference(input_data)
        elif self.is_llm:
            # Perform inference for LLM models
            return self.perform_llm_inference(input_data)

    def perform_vision_inference(self, input_data):
        """
        Perform inference on vision models.
        """
        infer_request = self.compiled_model.create_infer_request()
        infer_request.infer({self.compiled_model.input(0): input_data})
        return infer_request.get_output_tensor(self.compiled_model.output(0)).data

    def perform_llm_inference(self, input_data):
        """
        Perform inference on LLM models.
        """
        input_ids, attention_mask = (
            input_data  # Assuming input_data is (input_ids, attention_mask)
        )
        infer_request = self.compiled_model.create_infer_request()
        infer_request.infer(
            {
                self.compiled_model.input(0): input_ids,
                self.compiled_model.input(1): attention_mask,
            }
        )
        return infer_request.get_output_tensor(self.compiled_model.output(0)).data
