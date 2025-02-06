# TensorFlowEngine
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

import tensorflow as tf  # Standard TensorFlow
from .ml_engine import MLEngine


class TensorFlowEngine(MLEngine):
    def load_model(self, model_name, **kwargs):
        """Load a TensorFlow SavedModel or a model from a file."""
        try:
            # Load the TensorFlow model
            self.model = tf.saved_model.load(model_name)
            self.logger.info(f"TensorFlow model '{model_name}' loaded successfully.")
        except Exception as e:
            self.logger.error(
                f"Failed to load TensorFlow model '{model_name}'. Error: {e}"
            )

    def set_device(self, device):
        """Set the device for TensorFlow."""
        self.device = device
        self.logger.info(f"TensorFlow device set to {device}")

    def forward(self, frame):
        """
        Perform inference using the TensorFlow model.
        """
        if self.model is None:
            self.logger.error("No TensorFlow model loaded.")
            return None

        # Preprocess the frame (resize, normalize, etc.)
        input_tensor = tf.convert_to_tensor(frame, dtype=tf.float32)
        input_tensor = tf.expand_dims(
            input_tensor, 0
        )  # Add batch dimension if required

        # Run inference
        try:
            with tf.device(self.device):
                results = self.model(input_tensor)

            # Convert results to numpy arrays if necessary
            if isinstance(results, dict):
                results = {key: val.numpy() for key, val in results.items()}
            else:
                results = [val.numpy() for val in results]

            return results
        except Exception as e:
            self.logger.error(f"Error during TensorFlow inference: {e}")
            return None
