# PyTorchEngine
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

import gc
import os
import numpy as np
from PIL import Image
import torch
from torchvision import models
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoImageProcessor,
    VisionEncoderDecoderModel,
    AutoProcessor,
    BitsAndBytesConfig,
)

from .ml_engine import MLEngine


class PyTorchEngine(MLEngine):
    def load_model(self, model_name, **kwargs):
        """Load a pre-trained model by name from TorchVision, Transformers, or a local path."""
        processor_name = kwargs.get("processor_name")
        tokenizer_name = kwargs.get("tokenizer_name")

        try:
            # Special case for Phi-3-vision model from Hugging Face
            if model_name == "phi-3.5-vision":
                quantization_config = BitsAndBytesConfig(load_in_4bit=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    "microsoft/Phi-3.5-vision-instruct",
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    _attn_implementation="flash_attention_2",
                )
                self.processor = AutoProcessor.from_pretrained(
                    "microsoft/Phi-3.5-vision-instruct", trust_remote_code=True
                )
                self.logger.info(
                    "Phi-3.5-vision model and processor loaded successfully."
                )
                self.vision_language_model = True
                self.model.eval()
            elif os.path.isfile(model_name):
                self.model = torch.load(model_name)
                self.logger.info(f"Model loaded from local path: {model_name}")
            else:
                if hasattr(models, model_name):
                    self.model = getattr(models, model_name)(pretrained=True)
                    self.logger.info(
                        f"Pre-trained vision model '{model_name}' loaded from TorchVision"
                    )
                elif hasattr(models.detection, model_name):
                    self.model = getattr(models.detection, model_name)(
                        weights="DEFAULT"
                    )
                    self.logger.info(
                        f"Pre-trained detection model '{model_name}' loaded from TorchVision.detection"
                    )
                elif processor_name and tokenizer_name:
                    self.image_processor = AutoImageProcessor.from_pretrained(
                        processor_name
                    )
                    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                    self.set_model(
                        VisionEncoderDecoderModel.from_pretrained(model_name)
                    )
                    self.frame_stride = self.model.config.encoder.num_frames
                    self.logger.info(
                        f"Vision-Text model '{model_name}' loaded with processor and tokenizer."
                    )
                else:
                    self.logger.info(
                        f"Loading tokenizer for language model {model_name}"
                    )
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.logger.info(f"Loading language model {model_name}")
                    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
                    self.set_model(
                        AutoModelForCausalLM.from_pretrained(
                            model_name,
                            torch_dtype=(
                                torch.float16
                                if self.device == "cuda"
                                else torch.float32
                            ),
                            device_map="auto",
                            quantization_config=quantization_config,
                        )
                    )
                    self.get_model().eval()
                    self.logger.info(
                        f"Pre-trained LLM model '{model_name}' loaded from Transformers."
                    )

            self.execute_with_stream(lambda: self.model.to(self.device))
            self.logger.info(f"Model moved to {self.device}")

            return True

        except Exception as e:
            self.logger.error(f"Error loading model '{model_name}': {e}")
            self.tokenizer = None
            self.model = None
            return False

    def set_device(self, device):
        """Set PyTorch device for the model."""
        self.device = device
        self.logger.info(f"Setting device to {device}")

        if "cuda" in device:
            if not torch.cuda.is_available():
                self.logger.warning("CUDA is not available, falling back to CPU")
                self.device = "cpu"
                if self.model:
                    try:
                        self.model = self.model.cpu()
                        self.logger.info("Model moved to CPU due to unavailable CUDA")
                    except Exception as e:
                        self.logger.error(f"Failed to move model to CPU: {e}")
                return

            try:
                # Extract device index (e.g., "cuda:0" -> "0")
                self.device_index = device.split(":")[-1] if ":" in device else "0"
                torch.cuda.set_device(int(self.device_index))
                self.logger.info(f"CUDA device set to cuda:{self.device_index}")

                # Model placement is handled by device_map="auto" in load_model
                # Only move model if it exists and is not already on the correct device
                if self.model:
                    current_devices = {
                        param.device for param in self.model.parameters()
                    }
                    if any(d.type != "cuda" for d in current_devices):
                        self.logger.warning(
                            f"Model tensors found on {current_devices}, moving to {device}"
                        )
                        try:
                            self.model = self.model.to(device)
                            self.logger.info(f"Model moved to {device}")
                        except Exception as e:
                            self.logger.error(f"Failed to move model to {device}: {e}")
                            self.logger.warning("Falling back to CPU")
                            self.device = "cpu"
                            self.model = self.model.cpu()
            except Exception as e:
                self.logger.error(f"Failed to set CUDA device: {e}")
                self.logger.warning("Falling back to CPU")
                self.device = "cpu"
                if self.model:
                    self.model = self.model.cpu()

        elif device == "cpu":
            if self.model:
                try:
                    if not any(p.is_meta for p in self.model.parameters()):
                        self.model = self.model.cpu()
                        self.logger.info(f"Model moved to CPU")
                    else:
                        self.logger.error(
                            "Model contains meta tensors, cannot move to CPU"
                        )
                except Exception as e:
                    self.logger.error(f"Error moving model to CPU: {e}")

        else:
            self.logger.error(f"Invalid device specified: {device}")
            self.device = "cpu"
            if self.model:
                self.model = self.model.cpu()

    def _forward_classification(self, frames):
        """Handle inference for classification models like ResNet."""
        self.model.eval()
        is_batch = frames.ndim == 4  # (B, H, W, C) vs (H, W, C)
        img_tensor = (
            torch.from_numpy(np.array(frames, copy=True))
            .permute(
                0 if is_batch else 2,
                1 if is_batch else 0,
                2 if is_batch else 1,
                3 if is_batch else None,
            )
            .float()
        )
        img_tensor /= 255.0
        if not is_batch:
            img_tensor = img_tensor.unsqueeze(0)  # Add batch dim for single frame
        img_tensor = img_tensor.to(self.device)

        with torch.inference_mode():
            results = self.model(img_tensor)
        return (
            results.squeeze() if not is_batch else results
        )  # Remove batch dim if single

    def forward(self, frames):
        """Handle inference for different types of models, supporting single frames or batches."""
        is_batch = isinstance(frames, np.ndarray) and frames.ndim == 4  # (B, H, W, C)
        if not isinstance(frames, (np.ndarray, str)):
            self.logger.error(f"Invalid input type for forward: {type(frames)}")
            return None

        if self.vision_language_model and self.processor:
            try:
                from PIL import Image
                import torch
                import gc

                # Convert frames to PIL images
                if is_batch:
                    images = [Image.fromarray(np.uint8(frame)) for frame in frames]
                else:
                    images = [Image.fromarray(np.uint8(frames))]

                # Create prompt with placeholders for all images
                prompt_content = (
                    "\n".join([f"<|image_{i+1}|>" for i in range(len(images))])
                    + f"\n{self.prompt}"
                )
                messages = [{"role": "user", "content": prompt_content}]
                prompt = self.processor.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                # Process inputs for batch inference
                inputs = self.processor(prompt, images, return_tensors="pt").to(
                    self.device
                )

                # Run inference
                generation_args = {
                    "max_new_tokens": 500,
                    "temperature": 0.0,
                    "do_sample": False,
                }
                with torch.inference_mode():
                    generate_ids = self.model.generate(
                        **inputs,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                        **generation_args,
                    )

                # Decode response
                generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
                response = self.processor.batch_decode(
                    generate_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

                # Split response into per-frame captions (adjust based on model output)
                if is_batch:
                    # Assume response contains captions separated by newlines or repeated
                    captions = (
                        response.split("\n")[: len(images)]
                        if "\n" in response
                        else [response] * len(images)
                    )
                else:
                    captions = [response]

                self.logger.info(f"Generated captions: {captions}")

                # Clean up
                del inputs, generate_ids
                torch.cuda.empty_cache()
                gc.collect()

                return captions if is_batch else captions[0]

            except Exception as e:
                self.logger.error(f"Vision-language inference error: {e}")
                return None

        elif self.image_processor and self.tokenizer:
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
                    ).pixel_values.to(self.device)
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

        elif not self.tokenizer:
            self.model.eval()
            if "resnet" in self.model.__class__.__name__.lower():
                preds = self._forward_classification(frames)
                preds = (
                    preds.cpu().numpy() if isinstance(preds, torch.Tensor) else preds
                )
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

            # Detection models (e.g., Mask R-CNN) with true batch inference
            writable_frames = np.array(frames, copy=True)
            img_tensor = torch.from_numpy(writable_frames).float() / 255.0
            if is_batch:
                img_tensor = img_tensor.permute(
                    0, 3, 1, 2
                )  # (B, H, W, C) -> (B, C, H, W)
            else:
                img_tensor = img_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
                img_tensor = img_tensor.unsqueeze(0)  # Add batch dim: (1, C, H, W)
            img_tensor = img_tensor.to(self.device)

            with torch.inference_mode():
                results = self.model(img_tensor)  # Batch inference

            # Convert results to NumPy for consistency
            output_np = [
                {
                    k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                    for k, v in res.items()
                }
                for res in results
            ]
            self.logger.debug(
                f"Batch inference results: {len(output_np)} frames processed"
            )
            return output_np[0] if not is_batch else output_np

        elif self.tokenizer and not self.image_processor:
            if is_batch:
                self.logger.error("Batch processing not supported for LLM-only models.")
                return None
            inputs = self.tokenizer(frames, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                generated_tokens = self.model.generate(**inputs)
            generated_text = self.tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
            self.logger.info(f"Generated text: {generated_text}")
            return generated_text

        else:
            raise ValueError("Unsupported model type or missing processor/tokenizer.")

    def generate(self, input_text, max_length=100):
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_length=max_length)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.logger.info(f"Generated text: {generated_text}")
        return generated_text

    def execute_with_stream(self, func, *args, **kwargs):
        if self.device_queue_id is not None and "cuda" in self.device:
            s = torch.cuda.Stream(
                device=self.device, priority=0, stream_id=self.device_queue_id
            )
            with torch.cuda.stream(s):
                return func(*args, **kwargs)
        return func(*args, **kwargs)
