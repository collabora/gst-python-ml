# ONNXEngine
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
import onnxruntime as ort  # ONNX Runtime for executing ONNX models
from .ml_engine import MLEngine


class ONNXEngine(MLEngine):
    def __init__(self, device="cpu"):
        """
        Initialize the ONNX engine with the specified device.
        """
        super().__init__(device)
        self.session = None
        self.input_names = None
        self.output_names = None

    def load_model(self, model_name, **kwargs):
        """
        Load the ONNX model from the specified file path.
        """
        try:
            # Create the ONNX runtime session with device (CPU, CUDA, etc.)
            providers = (
                ["CPUExecutionProvider"]
                if self.device == "cpu"
                else ["CUDAExecutionProvider"]
            )
            self.session = ort.InferenceSession(model_name, providers=providers)

            # Extract input and output names for reference
            self.input_names = [inp.name for inp in self.session.get_inputs()]
            self.output_names = [out.name for out in self.session.get_outputs()]

            self.logger.info(
                f"ONNX model '{model_name}' loaded successfully on {self.device}."
            )
        except Exception as e:
            self.logger.error(f"Failed to load ONNX model '{model_name}'. Error: {e}")

    def set_device(self, device):
        """
        Set the device for inference.
        """
        self.device = device
        # ONNX Runtime does not allow changing the device after session creation,
        # so we need to reload the model if device changes.
        if self.session:
            model_path = (
                self.session.get_modelmeta().producer_name
            )  # Assuming model path is stored
            self.load_model(model_path)

    def forward(self, frame):
        """
        Perform inference on the given frame using the ONNX model.
        """
        try:
            # Preprocess the frame (resize, normalize, etc.) as required by the model
            input_tensor = np.expand_dims(
                frame.astype(np.float32), axis=0
            )  # Add batch dimension

            # Prepare the input dictionary
            input_dict = {self.input_names[0]: input_tensor}

            # Run inference
            output_data = self.session.run(self.output_names, input_dict)

            # Convert output to NumPy arrays if needed
            results = [np.array(out) for out in output_data]

            return (
                results if len(results) > 1 else results[0]
            )  # Return single result or list of results

        except Exception as e:
            self.logger.error(f"Error during ONNX inference: {e}")
            return None
