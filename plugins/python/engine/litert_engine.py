# LiteRTEngine
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


class LiteRTEngine(MLEngine):
    def __init__(self):
        super().__init__()
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.delegate = None
        self.model_name = None
        self.kwargs = None
        self.model_type = None

    def _create_delegate(self, delegate_path):
        try:
            delegate = tf.lite.experimental.load_delegate(delegate_path)
            self.logger.info(f"Delegate loaded successfully from '{delegate_path}'")
            return delegate
        except Exception as e:
            self.logger.error(f"Failed to load delegate from '{delegate_path}': {e}")
            return None

    def do_load_model(self, model_name, **kwargs):
        """Load a pre-trained model and convert to TFLite if necessary."""
        processor_name = kwargs.get("processor_name")
        tokenizer_name = kwargs.get("tokenizer_name")
        self.model_name = model_name
        self.kwargs = kwargs

        if os.path.isfile(model_name) and model_name.endswith(".tflite"):
            self.interpreter = tf.lite.Interpreter(
                model_path=model_name,
                experimental_delegates=[self.delegate] if self.delegate else None,
            )
            self.model_type = "custom"
            self.logger.info(f"TFLite model loaded from local path: {model_name}")
        else:
            if hasattr(keras.applications, model_name):
                model = getattr(keras.applications, model_name)(weights="imagenet")
                self.model_type = "classification"
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
                model = TFVisionEncoderDecoderModel.from_pretrained(model_name)
                self.frame_stride = (
                    model.config.encoder.num_frames
                    if hasattr(model.config.encoder, "num_frames")
                    else 1
                )
                self.model_type = "vision_text"
                self.logger.warning(
                    "Vision-text models in TFLite may require custom generation loops."
                )
            else:
                from transformers import AutoTokenizer, TFAutoModelForCausalLM

                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = TFAutoModelForCausalLM.from_pretrained(model_name)
                self.model_type = "llm"
                self.logger.warning(
                    "LLM models in TFLite require manual generation loops for inference."
                )

            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            tflite_model = converter.convert()

            self.interpreter = tf.lite.Interpreter(
                model_content=tflite_model,
                experimental_delegates=[self.delegate] if self.delegate else None,
            )
            self.logger.info(f"Model '{model_name}' converted to TFLite and loaded.")

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        return True

    def do_set_device(self, device):
        """Set the device/delegate for TFLite."""
        self.device = device
        self.logger.info(f"Setting device to {device}")

        if "cpu" in device:
            self.delegate = None
        elif "cuda" in device.lower() or device.lower() == "gpu":
            # TFLite has no CUDA delegate on desktop — fall back to CPU
            self.logger.warning(
                "TFLite has no CUDA/GPU delegate on desktop, falling back to CPU"
            )
            self.device = "cpu"
            self.delegate = None
        else:
            # Treat value as a path to a delegate .so
            self.delegate = self._create_delegate(device)

        # Reload the model with the new delegate
        if self.model_name:
            self.do_load_model(self.model_name, **self.kwargs)

    def _forward_classification(self, frames):
        """Handle inference for classification models."""
        is_batch = frames.ndim == 4
        input_shape = self.input_details[0]["shape"]
        if is_batch:
            batch_size = frames.shape[0]
            new_shape = [batch_size] + list(input_shape[1:])
            self.interpreter.resize_tensor_input(
                self.input_details[0]["index"], new_shape
            )
            self.interpreter.allocate_tensors()
        else:
            frames = np.expand_dims(frames, axis=0)

        img_array = frames.astype(np.float32) / 255.0
        self.interpreter.set_tensor(self.input_details[0]["index"], img_array)

        self.interpreter.invoke()

        preds = self.interpreter.get_tensor(self.output_details[0]["index"])
        probs = np.exp(preds) / np.sum(np.exp(preds), axis=1, keepdims=True)
        top_classes = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        results = [
            {"labels": [int(c)], "scores": [float(s)]}
            for c, s in zip(top_classes, confidences)
        ]
        return results[0] if not is_batch else results

    def do_forward(self, frames):
        """Handle inference for different types of models, supporting single frames or batches."""
        is_batch = isinstance(frames, np.ndarray) and frames.ndim == 4
        if not isinstance(frames, (np.ndarray, str)):
            self.logger.error(f"Invalid input type for forward: {type(frames)}")
            return None

        if self.model_type == "vision_text":
            # Basic support; may need custom loop for generation
            if is_batch:
                self.logger.error(
                    "Batch processing not supported for vision-text models."
                )
                return None
            if not hasattr(self, "counter"):
                self.counter = 0
                self.frame_buffer = []
            self.counter += 1
            if self.counter % self.frame_stride == 0:
                self.frame_buffer.append(frames)
            if len(self.frame_buffer) >= self.batch_size:
                pixel_values = self.image_processor(
                    self.frame_buffer, return_tensors="tf"
                ).pixel_values
                # Assuming model exported for single forward pass; adjust if needed
                input_shape = self.input_details[0]["shape"]
                if pixel_values.shape != tuple(input_shape):
                    self.logger.error("Input shape mismatch for vision-text model.")
                    return None
                self.interpreter.set_tensor(
                    self.input_details[0]["index"], pixel_values.numpy()
                )
                self.interpreter.invoke()
                tokens = self.interpreter.get_tensor(self.output_details[0]["index"])
                captions = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
                self.frame_buffer = []
                return captions[0]
            return None

        elif self.model_type == "llm":
            if is_batch:
                self.logger.error("Batch processing not supported for LLM models.")
                return None
            # For LLMs, forward might not be directly applicable; use generate instead
            self.logger.warning("Use generate method for LLM inference.")
            return None

        elif self.model_type == "classification":
            return self._forward_classification(frames)

        else:  # Assume detection or custom
            input_shape = self.input_details[0]["shape"]
            if self.input_format == "auto":
                # a litert export puts channels first, an onnx2tf export puts them last
                self.input_format = "nchw" if input_shape[1] in (1, 3, 4) else "nhwc"
            img = self._apply_input_format(frames.astype(np.float32) / 255.0, is_batch)
            if is_batch:
                new_shape = [img.shape[0]] + list(input_shape[1:])
                self.interpreter.resize_tensor_input(
                    self.input_details[0]["index"], new_shape
                )
                self.interpreter.allocate_tensors()

            self.interpreter.set_tensor(self.input_details[0]["index"], img)
            self.interpreter.invoke()

            outputs = [
                self.interpreter.get_tensor(output["index"])
                for output in self.output_details
            ]

            # Standard TFLite detection: [boxes, classes, scores, num_detections]
            if len(outputs) >= 4:
                boxes, classes, scores, num_dets = outputs[:4]
                results = []
                for i in range(img.shape[0]):
                    n = int(num_dets[i])
                    res = {
                        "boxes": boxes[i][:n],
                        "labels": classes[i][:n].astype(int),
                        "scores": scores[i][:n],
                    }
                    results.append(res)
                return results[0] if not is_batch else results
            else:
                raw = outputs[0] if len(outputs) == 1 else outputs
                results = self._apply_post_process(raw, is_batch)
                return self._scaled_to_input(results, input_shape)

    # ultralytics tflite exports return boxes as fractions of the input size
    def _scaled_to_input(self, results, input_shape):
        if self.input_format == "nchw":
            height, width = input_shape[2], input_shape[3]
        else:
            height, width = input_shape[1], input_shape[2]
        for result in results if isinstance(results, list) else [results]:
            if not isinstance(result, dict):
                continue
            boxes = np.asarray(result["boxes"], dtype=np.float32)
            if len(boxes) == 0 or boxes.max() > 1.0:
                continue
            result["boxes"] = boxes * np.array(
                [width, height, width, height], dtype=np.float32
            )
        return results

    def do_generate(self, input_text, max_length=1000, system_prompt=None):
        if self.model_type != "llm":
            raise ValueError("Generate is only supported for LLM models.")
        # Basic greedy generation loop; assumes model exported as single-step autoregressive
        input_ids = self.tokenizer(input_text, return_tensors="np")["input_ids"]
        generated_ids = input_ids.copy()

        for _ in range(max_length):
            self.interpreter.set_tensor(
                self.input_details[0]["index"], generated_ids[:, -1:]
            )  # Last token
            self.interpreter.invoke()
            logits = self.interpreter.get_tensor(self.output_details[0]["index"])
            next_token = np.argmax(logits[:, -1, :], axis=-1)
            generated_ids = np.concatenate((generated_ids, next_token[:, None]), axis=1)
            if next_token[0] == self.tokenizer.eos_token_id:
                break

        generated_text = self.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )
        self.logger.info(f"Generated text: {generated_text}")
        return generated_text
