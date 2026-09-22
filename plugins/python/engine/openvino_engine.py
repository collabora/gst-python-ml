# OpenVinoEngine
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
import openvino as ov

from .ml_engine import MLEngine


class OpenVinoEngine(MLEngine):
    def __init__(self):
        super().__init__()
        self.core = ov.Core()
        self.compiled_model = None
        self.ov_model = None
        self.model_type = None
        self.model_name = None
        self.kwargs = None
        self.detection_threshold = 0.5

    def do_load_model(self, model_name, **kwargs):
        """Load a pre-trained model by name from TorchVision, Transformers (via Optimum OpenVINO), or a local IR path."""
        processor_name = kwargs.get("processor_name")
        tokenizer_name = kwargs.get("tokenizer_name")
        self.model_name = model_name
        self.kwargs = kwargs

        if model_name.endswith((".xml", ".bin")):
            base = model_name.rsplit(".", 1)[0]
        else:
            base = model_name
        xml_path = f"{base}.xml"
        bin_path = f"{base}.bin"
        if os.path.isfile(xml_path) and os.path.isfile(bin_path):
            self.ov_model = self.core.read_model(xml_path)
            self.model_type = "custom"
            self.logger.info(f"OpenVINO IR model loaded from local path: {model_name}")
        else:
            from torchvision import models

            if hasattr(models, model_name):
                pt_model = getattr(models, model_name)(pretrained=True)
                self.ov_model = ov.convert_model(pt_model)
                self.model_type = "classification"
                self.logger.info(
                    f"Pre-trained vision model '{model_name}' converted to OpenVINO."
                )
            elif hasattr(models.detection, model_name):
                pt_model = getattr(models.detection, model_name)(pretrained=True)
                self.ov_model = ov.convert_model(pt_model)
                self.model_type = "detection"
                self.logger.info(
                    f"Pre-trained detection model '{model_name}' converted to OpenVINO."
                )
            elif processor_name and tokenizer_name:
                from transformers import AutoTokenizer, AutoImageProcessor
                from optimum.intel import OVModelForVision2Seq

                self.image_processor = AutoImageProcessor.from_pretrained(
                    processor_name
                )
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                self.compiled_model = OVModelForVision2Seq.from_pretrained(
                    model_name, export=True
                )
                self.frame_stride = (
                    self.compiled_model.config.encoder.num_frames
                    if hasattr(self.compiled_model.config.encoder, "num_frames")
                    else 1
                )
                self.model_type = "vision_text"
                self.logger.info(
                    f"Vision-Text model '{model_name}' loaded via Optimum OpenVINO."
                )
                return True
            else:
                from transformers import AutoTokenizer
                from optimum.intel import OVModelForCausalLM

                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.compiled_model = OVModelForCausalLM.from_pretrained(
                    model_name, export=True
                )
                self.model_type = "llm"
                self.logger.info(
                    f"Pre-trained LLM model '{model_name}' loaded via Optimum OpenVINO."
                )
                return True

        self.compiled_model = self.core.compile_model(self.ov_model, self.device)
        self.logger.info(f"Model compiled on {self.device}")

        return True

    def do_set_device(self, device):
        """Set OpenVINO device for the model."""
        # Map cuda/gpu aliases to OpenVINO device names (uppercase)
        if "cuda" in device.lower() or device.lower() == "gpu":
            self.device = "GPU"
        else:
            self.device = device.upper()
        self.logger.info(f"Setting device to {self.device}")

        available_devices = self.core.available_devices
        if self.device not in available_devices:
            if "GPU" in self.device and any("GPU" in d for d in available_devices):
                self.device = "GPU"
            else:
                self.logger.warning(
                    f"Device {self.device} not available, falling back to CPU"
                )
                self.device = "CPU"

        # Recompile model if already loaded
        if self.model_name:
            self.do_load_model(self.model_name, **self.kwargs)

    def _forward_classification(self, frames):
        """Handle inference for classification models."""
        is_batch = frames.ndim == 4
        img_array = np.array(frames, dtype=np.float32) / 255.0
        if is_batch:
            img_array = np.transpose(
                img_array, (0, 3, 1, 2)
            )  # (B, H, W, C) -> (B, C, H, W)
        else:
            img_array = np.transpose(img_array, (2, 0, 1))  # (H, W, C) -> (C, H, W)
            img_array = np.expand_dims(img_array, 0)

        infer_request = self.compiled_model.create_infer_request()
        infer_request.infer({0: img_array})
        preds = infer_request.get_output_tensor(0).data
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
                gen_kwargs = {"min_length": 10, "max_length": 20, "num_beams": 8}
                pixel_values = self.image_processor(
                    self.frame_buffer, return_tensors="np"
                ).pixel_values
                tokens = self.compiled_model.generate(pixel_values, **gen_kwargs)
                captions = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
                self.logger.info(f"Captions: {captions}")
                self.frame_buffer = []
                return captions[0]
            return None

        elif self.model_type == "llm":
            if is_batch:
                self.logger.error("Batch processing not supported for LLM-only models.")
                return None
            inputs = self.tokenizer(frames, return_tensors="np")
            generated_tokens = self.compiled_model.generate(**inputs)
            generated_text = self.tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
            self.logger.info(f"Generated text: {generated_text}")
            return generated_text

        elif self.model_type == "classification":
            return self._forward_classification(frames)

        elif self.model_type == "detection":
            writable_frames = np.array(frames, copy=True, dtype=np.float32) / 255.0
            if is_batch:
                img_array = np.transpose(
                    writable_frames, (0, 3, 1, 2)
                )  # (B, H, W, C) -> (B, C, H, W)
            else:
                img_array = np.transpose(writable_frames, (2, 0, 1))
                img_array = np.expand_dims(img_array, 0)
            infer_request = self.compiled_model.create_infer_request()
            infer_request.infer({0: img_array})
            outputs = [
                infer_request.get_output_tensor(i).data
                for i in range(len(self.compiled_model.outputs))
            ]
            # Assuming outputs for detection: [boxes, labels, scores]
            if len(outputs) == 3:
                boxes, labels, scores = outputs
                results = []
                for i in range(img_array.shape[0]):
                    valid = scores[i] > self.detection_threshold
                    res = {
                        "boxes": boxes[i][valid],
                        "labels": labels[i][valid].astype(int),
                        "scores": scores[i][valid],
                    }
                    results.append(res)
            else:
                raise ValueError("Unexpected output format for detection model.")
            self.logger.debug(
                f"Batch inference results: {len(results)} frames processed"
            )
            return results[0] if not is_batch else results

        elif self.model_type == "custom":
            # Generic forward for custom models
            img = self._apply_input_format(frames.astype(np.float32) / 255.0, is_batch)
            infer_request = self.compiled_model.create_infer_request()
            infer_request.infer({0: img})
            outputs = [
                infer_request.get_output_tensor(i).data
                for i in range(len(self.compiled_model.outputs))
            ]
            raw = outputs if len(outputs) > 1 else outputs[0]
            return self._apply_post_process(raw, is_batch)

        else:
            raise ValueError("Unsupported model type.")

    def do_generate(self, input_text, max_length=1000, system_prompt=None):
        if self.model_type != "llm":
            raise ValueError("Generate is only supported for LLM models.")
        inputs = self.tokenizer(input_text, return_tensors="np")
        outputs = self.compiled_model.generate(**inputs, max_length=max_length)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.logger.info(f"Generated text: {generated_text}")
        return generated_text
