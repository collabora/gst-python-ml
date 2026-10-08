# TensorFlowEngine
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

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

from .ml_engine import MLEngine
from .onnx_to_tensorflow import (
    ONNX_SUFFIX,
    onnx_output_names,
    saved_model_from_onnx,
)


class TensorFlowEngine(MLEngine):
    def do_load_model(self, model_name, **kwargs):
        self.model_type = None
        self.output_names = None

        if model_name.endswith(ONNX_SUFFIX):
            self.output_names = onnx_output_names(model_name)
            model_name = str(saved_model_from_onnx(model_name))

        if os.path.isdir(model_name):
            self.model = tf.saved_model.load(model_name)
            self.infer = self.model.signatures["serving_default"]
            self.model_type = "custom"
            self.logger.info(f"SavedModel loaded from local path: {model_name}")
        elif os.path.isfile(model_name):
            self.model = tf.keras.models.load_model(model_name)
            self.model_type = "custom"
            self.logger.info(f"Keras model loaded from local path: {model_name}")
        else:
            if hasattr(keras.applications, model_name):
                self.model = getattr(keras.applications, model_name)(weights="imagenet")
                self.model_type = "classification"
                self.logger.info(
                    f"Pre-trained vision model '{model_name}' loaded from keras.applications"
                )
            else:
                raise FileNotFoundError(
                    "TensorFlow takes a SavedModel directory, a Keras file or a "
                    f"keras.applications name, got: {model_name}"
                )

        if hasattr(self.model, "trainable"):
            self.model.trainable = False
        return True

    def do_set_device(self, device):
        """Set TensorFlow device for the model."""
        self.device = device
        self.logger.info(f"Setting device to {device}")

        if device == "cpu":
            self.device = "/cpu:0"
            tf.config.set_visible_devices([], "GPU")
        elif "cuda" in device:
            gpus = tf.config.list_physical_devices("GPU")
            if not gpus:
                raise RuntimeError(f"TensorFlow sees no GPU for device={device}")
            index = int(device.split(":")[-1]) if ":" in device else 0
            tf.config.set_visible_devices(gpus[index], "GPU")
        else:
            raise ValueError(f"Invalid device specified: {device}")

    def _forward_classification(self, frames):
        """Handle inference for classification models like ResNet."""
        is_batch = frames.ndim == 4  # (B, H, W, C) vs (H, W, C)
        img_tensor = tf.convert_to_tensor(frames, dtype=tf.float32)
        img_tensor /= 255.0
        if not is_batch:
            img_tensor = tf.expand_dims(img_tensor, 0)  # Add batch dim for single frame

        with tf.device(self.device):
            results = self.model(img_tensor, training=False)
        return results[0] if not is_batch else results  # Remove batch dim if single

    def do_forward(self, frames):
        """Handle inference for different types of models, supporting single frames or batches."""
        is_batch = isinstance(frames, np.ndarray) and frames.ndim == 4  # (B, H, W, C)
        if not isinstance(frames, np.ndarray):
            self.logger.error(f"Invalid input type for forward: {type(frames)}")
            return None

        if self.model_type == "classification":
            preds = self._forward_classification(frames)
            preds = preds.numpy() if isinstance(preds, tf.Tensor) else preds
            if not is_batch:
                preds = np.expand_dims(preds, 0)
            results = self._top_classes(preds, is_batch)
            self.logger.info(f"Classification results: {results}")
            return results

        else:
            # General models (e.g., detection or custom) with true batch inference
            img = self._apply_input_format(
                np.array(frames, copy=True, dtype=np.float32) / 255.0, is_batch
            )
            img_tensor = tf.convert_to_tensor(img)

            with tf.device(self.device):
                if hasattr(self, "infer"):
                    results = self.infer(img_tensor)
                else:
                    results = self.model(img_tensor, training=False)

            # Convert results to NumPy for consistency
            if isinstance(results, dict) and self.output_names:
                outputs = [results[name].numpy() for name in self.output_names]
                output_np = outputs if len(outputs) > 1 else outputs[0]
            elif isinstance(results, dict):
                output_np = {
                    k: v.numpy() if isinstance(v, tf.Tensor) else v
                    for k, v in results.items()
                }
                if len(output_np) == 1:
                    output_np = next(iter(output_np.values()))
            elif isinstance(results, (list, tuple)):
                output_np = [
                    v.numpy() if isinstance(v, tf.Tensor) else v for v in results
                ]
            else:
                output_np = (
                    results.numpy() if isinstance(results, tf.Tensor) else results
                )
            self.logger.debug(
                f"Batch inference results: {1 if not is_batch else len(frames)} frames processed"
            )
            return self._apply_post_process(output_np, is_batch)

    def do_generate(self, input_text, max_length=1000, system_prompt=None):
        raise NotImplementedError(
            "TensorFlow does not support text generation. "
            "Use PyTorch or llama.cpp for LLM workloads."
        )
