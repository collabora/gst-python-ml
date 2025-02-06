# LiteRTEngine
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
import tensorflow as tf  # TensorFlow Lite interpreter
from .ml_engine import MLEngine


class LiteRTEngine(MLEngine):
    def __init__(self, device="cpu"):
        """
        Initializes the TFLite engine and attempts to load the delegate if provided.
        """
        super().__init__(device)
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.delegate = None

        # Try to load the delegate during initialization
        if device is not None and device != "cpu":
            self.delegate = self._create_delegate()

    def _create_delegate(self):
        """Creates a TFLite delegate from the provided path."""
        try:
            delegate = tf.lite.experimental.load_delegate(self.device)
            self.logger.info(f"Delegate loaded successfully from '{self.device}'")
            return delegate
        except Exception as e:
            self.logger.error(
                f"Failed to load delegate from '{self.device}'. Error: {e}"
            )
            return None  # Fall back to no delegate if loading fails

    def load_model(self, model_name, **kwargs):
        """Load a TFLite model from a file path."""
        try:
            # Load the TFLite model and allocate tensors, pass the delegate if available
            self.interpreter = tf.lite.Interpreter(
                model_path=model_name,
                experimental_delegates=[self.delegate] if self.delegate else None,
            )
            self.interpreter.allocate_tensors()

            # Get input and output tensor details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            self.logger.info(f"TFLite model '{model_name}' loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load TFLite model '{model_name}'. Error: {e}")

    def set_device(self, device):
        """TFLite does not require explicit device management in Python, but delegates handle this."""
        self.device = device

    def forward(self, frame):
        """
        Perform object detection using the TFLite model.
        """
        # Preprocess the frame (resize, normalize, etc.)
        input_shape = self.input_details[0]["shape"]
        frame_resized = np.resize(frame, input_shape).astype(np.float32)

        # Set the input tensor
        self.interpreter.set_tensor(self.input_details[0]["index"], frame_resized)

        # Run inference
        self.interpreter.invoke()

        # Get the output tensor(s)
        output_data = [
            self.interpreter.get_tensor(output["index"])
            for output in self.output_details
        ]

        # Convert all outputs to NumPy arrays (if not already)
        results = [np.array(data) for data in output_data]

        # If there's only one result, return it directly
        if len(results) == 1:
            return results[0]

        # Otherwise, return all results as a list
        return results
