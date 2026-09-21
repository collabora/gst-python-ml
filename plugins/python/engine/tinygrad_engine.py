# TinyGradEngine
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

import numpy as np

from .ml_engine import MLEngine


class TinyGradEngine(MLEngine):
    def __init__(self):
        super().__init__()
        self.model_type = None
        self.model_name = None
        self.kwargs = None

    def do_load_model(self, model_name, **kwargs):
        """Load a torchvision resnet into TinyGrad, or a Transformers model."""

        processor_name = kwargs.get("processor_name")
        tokenizer_name = kwargs.get("tokenizer_name")
        self.model_name = model_name
        self.kwargs = kwargs

        try:
            # TorchVision models
            from torchvision import models as tv_models

            if hasattr(tv_models, model_name):
                pt_model = getattr(tv_models, model_name)(pretrained=True)
                if not isinstance(pt_model, tv_models.ResNet):
                    self.logger.error(
                        f"TinyGrad runs the torchvision resnet family, not '{model_name}'."
                    )
                    return False
                from .tinygrad_resnet import tinygrad_resnet

                self.model = tinygrad_resnet(pt_model.eval())
                self.model_type = "classification"
                self.logger.info(
                    f"Pre-trained vision model '{model_name}' loaded with TinyGrad."
                )
                return True

            # Vision-text models via Transformers
            if processor_name and tokenizer_name:
                from transformers import (
                    AutoTokenizer,
                    AutoImageProcessor,
                    AutoModelForVision2Seq,
                )

                self.image_processor = AutoImageProcessor.from_pretrained(
                    processor_name
                )
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                pt_model = AutoModelForVision2Seq.from_pretrained(model_name)
                self.model = pt_model
                self.frame_stride = (
                    pt_model.config.encoder.num_frames
                    if hasattr(pt_model.config, "encoder")
                    and hasattr(pt_model.config.encoder, "num_frames")
                    else 1
                )
                self.model_type = "vision_text"
                self.logger.info(
                    f"Vision-Text model '{model_name}' loaded for TinyGrad engine."
                )
                return True

            # LLM models via Transformers
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model_type = "llm"
            self.logger.info(
                f"Pre-trained LLM model '{model_name}' loaded for TinyGrad engine."
            )
            return True

        except Exception as e:
            self.logger.error(f"Error loading model '{model_name}': {e}")
            self.tokenizer = None
            self.image_processor = None
            self.model = None
            return False

    def do_set_device(self, device):
        """Set TinyGrad device for the model."""
        from tinygrad import Device
        from tinygrad.helpers import DEV

        self.device = device
        self.logger.info(f"Setting device to {device}")

        device_upper = device.upper() if device else "CPU"
        try:
            Device[device_upper]
            DEV.value = device_upper
        except Exception:
            self.logger.warning(
                f"Device '{device}' not available in TinyGrad, falling back to CPU"
            )
            self.device = "cpu"
            DEV.value = "CPU"

    def _forward_classification(self, frames):
        """Handle inference for classification models."""

        is_batch = frames.ndim == 4
        img_array = np.array(frames, dtype=np.float32) / 255.0
        if is_batch:
            img_array = np.transpose(img_array, (0, 3, 1, 2))
        else:
            img_array = np.transpose(img_array, (2, 0, 1))
            img_array = np.expand_dims(img_array, 0)

        from tinygrad import Tensor

        preds = self.model(Tensor(img_array)).numpy()
        probs = np.exp(preds) / np.sum(np.exp(preds), axis=1, keepdims=True)
        top_classes = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        results = [
            {"labels": [int(c)], "scores": [float(s)]}
            for c, s in zip(top_classes, confidences)
        ]
        return results[0] if not is_batch else results

    def do_forward(self, frames):
        """Execute inference on a single frame or batch of frames."""
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
            self.counter += 1
            if self.counter % self.frame_stride == 0:
                self.frame_buffer.append(frames)
            if len(self.frame_buffer) >= self.batch_size:
                self.logger.info(f"Processing {self.batch_size} frames")
                try:
                    gen_kwargs = {"min_length": 10, "max_length": 20, "num_beams": 8}
                    pixel_values = self.image_processor(
                        self.frame_buffer, return_tensors="pt"
                    ).pixel_values
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
            inputs = self.tokenizer(frames, return_tensors="pt")
            generated_tokens = self.model.generate(**inputs)
            generated_text = self.tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
            self.logger.info(f"Generated text: {generated_text}")
            return generated_text

        elif self.model_type == "classification":
            return self._forward_classification(frames)

        else:
            raise ValueError("Unsupported model type.")

    def do_generate(self, input_text, max_length=1000, system_prompt=None):
        if self.model_type != "llm":
            raise ValueError("Generate is only supported for LLM models.")

        inputs = self.tokenizer(input_text, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=max_length)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.logger.info(f"Generated text: {generated_text}")
        return generated_text
