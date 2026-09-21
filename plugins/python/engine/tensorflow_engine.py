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


class TensorFlowEngine(MLEngine):
    def do_load_model(self, model_name, **kwargs):
        """Load a pre-trained model by name from keras.applications, Transformers, or a local path."""
        processor_name = kwargs.get("processor_name")
        tokenizer_name = kwargs.get("tokenizer_name")
        self.model_type = None

        try:
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
                    self.model = getattr(keras.applications, model_name)(
                        weights="imagenet"
                    )
                    self.model_type = "classification"
                    self.logger.info(
                        f"Pre-trained vision model '{model_name}' loaded from keras.applications"
                    )
                elif processor_name and tokenizer_name:
                    from transformers import (
                        AutoImageProcessor,
                        AutoTokenizer,
                        TFVisionEncoderDecoderModel,
                    )

                    self.image_processor = AutoImageProcessor.from_pretrained(
                        processor_name
                    )
                    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                    self.model = TFVisionEncoderDecoderModel.from_pretrained(model_name)
                    self.frame_stride = (
                        self.model.config.encoder.num_frames
                        if hasattr(self.model.config.encoder, "num_frames")
                        else 1
                    )
                    self.model_type = "vision_text"
                    self.logger.info(
                        f"Vision-Text model '{model_name}' loaded with processor and tokenizer."
                    )
                else:
                    from transformers import AutoTokenizer, TFAutoModelForCausalLM

                    self.logger.info(
                        f"Loading tokenizer for language model {model_name}"
                    )
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.logger.info(f"Loading language model {model_name}")
                    self.model = TFAutoModelForCausalLM.from_pretrained(
                        model_name,
                    )
                    self.model_type = "llm"
                    self.logger.info(
                        f"Pre-trained LLM model '{model_name}' loaded from Transformers."
                    )

            if hasattr(self.model, "trainable"):
                self.model.trainable = False
            return True

        except Exception as e:
            self.logger.error(f"Error loading model '{model_name}': {e}")
            self.tokenizer = None
            self.model = None
            return False

    def do_set_device(self, device):
        """Set TensorFlow device for the model."""
        self.device = device
        self.logger.info(f"Setting device to {device}")

        if "cuda" in device:
            gpus = tf.config.list_physical_devices("GPU")
            if not gpus:
                self.logger.warning("GPU is not available, falling back to CPU")
                self.device = "/cpu:0"
                tf.config.set_visible_devices([], "GPU")
                return

            try:
                index = int(device.split(":")[-1]) if ":" in device else 0
                tf.config.set_visible_devices(gpus[index], "GPU")
                self.logger.info(f"GPU device set to GPU:{index}")
            except Exception as e:
                self.logger.error(f"Failed to set GPU device: {e}")
                self.logger.warning("Falling back to CPU")
                self.device = "/cpu:0"
                tf.config.set_visible_devices([], "GPU")

        elif device == "cpu":
            self.device = "/cpu:0"
            tf.config.set_visible_devices([], "GPU")
            self.logger.info("Device set to CPU")

        else:
            self.logger.error(f"Invalid device specified: {device}")
            self.device = "/cpu:0"
            tf.config.set_visible_devices([], "GPU")

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
        if not isinstance(frames, (np.ndarray, str)):
            self.logger.error(f"Invalid input type for forward: {type(frames)}")
            return None

        if self.model_type == "vision_text":
            if is_batch:
                self.logger.error(
                    "Batch processing not supported for vision-text models with frame buffering."
                )
                return None
            if not hasattr(self, "counter"):
                self.counter = 0
                self.frame_buffer = []
            self.counter += 1
            if self.counter % self.frame_stride == 0:
                self.frame_buffer.append(frames)
            if len(self.frame_buffer) >= self.batch_size:
                self.logger.info(f"Processing {self.batch_size} frames")
                try:
                    gen_kwargs = {"min_length": 10, "max_length": 20, "num_beams": 8}
                    pixel_values = self.image_processor(
                        self.frame_buffer, return_tensors="tf"
                    ).pixel_values
                    with tf.device(self.device):
                        tokens = self.model.generate(pixel_values, **gen_kwargs)
                    captions = self.tokenizer.batch_decode(
                        tokens, skip_special_tokens=True
                    )
                    self.logger.info(f"Captions: {captions}")
                    self.frame_buffer = []
                    return captions[0]
                except Exception as e:
                    self.logger.error(f"Failed to process frames: {e}")
                    self.frame_buffer = []
                    return None
            return None

        elif self.model_type == "llm":
            if is_batch:
                self.logger.error("Batch processing not supported for LLM-only models.")
                return None
            inputs = self.tokenizer(frames, return_tensors="tf")
            with tf.device(self.device):
                generated_tokens = self.model.generate(**inputs)
            generated_text = self.tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
            self.logger.info(f"Generated text: {generated_text}")
            return generated_text

        elif self.model_type == "classification":
            preds = self._forward_classification(frames)
            preds = preds.numpy() if isinstance(preds, tf.Tensor) else preds
            if not is_batch:
                preds = np.expand_dims(preds, 0)
            probs = np.exp(preds) / np.sum(np.exp(preds), axis=1, keepdims=True)
            top_classes = np.argmax(probs, axis=1)
            confidences = np.max(probs, axis=1)
            results = [
                {"labels": [int(c)], "scores": [float(s)]}
                for c, s in zip(top_classes, confidences)
            ]
            self.logger.info(f"Classification results: {results}")
            return results[0] if not is_batch else results

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
            if isinstance(results, dict):
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
        inputs = self.tokenizer(input_text, return_tensors="tf")
        with tf.device(self.device):
            outputs = self.model.generate(**inputs, max_length=max_length)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.logger.info(f"Generated text: {generated_text}")
        return generated_text
