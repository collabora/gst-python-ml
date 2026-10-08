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
from ai_edge_litert.interpreter import Interpreter, load_delegate

from .ml_engine import MLEngine
from .onnx_to_tensorflow import (
    ONNX_SUFFIX,
    float32_tflite_from_onnx,
    onnx_output_names,
)


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
        self.output_names = None

    def do_load_model(self, model_name, **kwargs):
        """Load a pre-trained model and convert to TFLite if necessary."""
        self.model_name = model_name
        self.kwargs = kwargs
        self.output_names = None

        if model_name.endswith(ONNX_SUFFIX):
            self.output_names = onnx_output_names(model_name)
            model_name = str(float32_tflite_from_onnx(model_name))

        if os.path.isfile(model_name) and model_name.endswith(".tflite"):
            self.interpreter = Interpreter(
                model_path=model_name,
                experimental_delegates=[self.delegate] if self.delegate else None,
            )
            self.model_type = "custom"
            self.logger.info(f"TFLite model loaded from local path: {model_name}")
        else:
            # only converting a model needs tensorflow
            import tensorflow as tf
            from tensorflow import keras

            if hasattr(keras.applications, model_name):
                model = getattr(keras.applications, model_name)(weights="imagenet")
                self.model_type = "classification"
            else:
                raise FileNotFoundError(
                    "LiteRT takes a .tflite file or a keras.applications name, "
                    f"got: {model_name}"
                )

            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            tflite_model = converter.convert()

            self.interpreter = Interpreter(
                model_content=tflite_model,
                experimental_delegates=[self.delegate] if self.delegate else None,
            )
            self.logger.info(f"Model '{model_name}' converted to TFLite and loaded.")

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        if self.output_names:
            # the interpreter lists a converted model's outputs out of onnx order
            signature_outputs = (
                self.interpreter.get_signature_runner().get_output_details()
            )
            self.output_details = [
                signature_outputs[name] for name in self.output_names
            ]

        return True

    def do_set_device(self, device):
        """Set the device/delegate for TFLite."""
        if "cpu" in device:
            self.delegate = None
        elif "cuda" in device.lower() or device.lower() == "gpu":
            raise ValueError(
                f"LiteRT has no desktop GPU delegate for device={device}, "
                "use cpu or the path to a delegate library"
            )
        else:
            # Treat value as a path to a delegate .so
            self.delegate = load_delegate(device)
        self.device = device
        self.logger.info(f"Setting device to {device}")

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
        return self._top_classes(preds, is_batch)

    def do_forward(self, frames):
        """Handle inference for different types of models, supporting single frames or batches."""
        is_batch = isinstance(frames, np.ndarray) and frames.ndim == 4
        if not isinstance(frames, np.ndarray):
            self.logger.error(f"Invalid input type for forward: {type(frames)}")
            return None

        if self.model_type == "classification":
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
            if self.output_names is None and len(outputs) >= 4:
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
        raise NotImplementedError(
            "LiteRT does not support text generation. "
            "Use PyTorch or llama.cpp for LLM workloads."
        )
