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
)
from .ml_engine import MLEngine


class PyTorchEngine(MLEngine):
    def load_model(self, model_name, **kwargs):
        """Load a pre-trained model by name from TorchVision, Transformers, or a local path."""
        processor_name = kwargs.get("processor_name")
        tokenizer_name = kwargs.get("tokenizer_name")

        try:
            # Special case for Phi-3-vision model from Hugging Face
            if model_name == "phi-3-vision":
                self.model = AutoModelForCausalLM.from_pretrained(
                    "microsoft/Phi-3-vision-128k-instruct",
                    device_map="cuda",
                    trust_remote_code=True,
                    torch_dtype="auto",
                    _attn_implementation="flash_attention_2",
                ).to(self.device)
                self.processor = AutoProcessor.from_pretrained(
                    "microsoft/Phi-3-vision-128k-instruct", trust_remote_code=True
                )
                self.logger.info(
                    "Phi-3-vision model and processor loaded successfully."
                )
                self.vision_language_model = True
                self.model.eval()  # Set model to evaluation mode

            # Check if model_name is a valid file path (local model)
            elif os.path.isfile(model_name):
                # Load the model from the local path
                self.model = torch.load(model_name)
                self.logger.info(f"Model loaded from local path: {model_name}")

            else:
                # Check for TorchVision models
                if hasattr(models, model_name):
                    self.model = getattr(models, model_name)(pretrained=True)
                    self.logger.info(
                        f"Pre-trained vision model '{model_name}' loaded from TorchVision"
                    )

                # Check for TorchVision detection models
                elif hasattr(models.detection, model_name):
                    self.model = getattr(models.detection, model_name)(
                        weights="DEFAULT"
                    )
                    self.logger.info(
                        f"Pre-trained detection model '{model_name}' loaded from TorchVision.detection"
                    )

                # Handle Vision-Text models by passing processor and tokenizer names
                elif processor_name and tokenizer_name:
                    # Load the image processor (for vision inputs)
                    self.image_processor = AutoImageProcessor.from_pretrained(
                        processor_name
                    )
                    self.logger.info(
                        f"Image processor '{processor_name}' loaded successfully."
                    )

                    # Load the tokenizer (for text outputs)
                    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                    self.logger.info(
                        f"Tokenizer '{tokenizer_name}' loaded successfully."
                    )

                    # Load the vision-text model using VisionEncoderDecoderModel
                    self.set_model(
                        VisionEncoderDecoderModel.from_pretrained(model_name)
                    )
                    self.frame_stride = self.model.config.encoder.num_frames
                    self.logger.info(
                        f"Vision-Text model '{model_name}' loaded with processor and tokenizer."
                    )

                # Handle general Hugging Face models (LLMs)
                else:
                    # Assume this is an LLM model
                    self.set_device(self.device)
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.set_model(
                        AutoModelForCausalLM.from_pretrained(
                            model_name,
                            torch_dtype=(
                                torch.float16
                                if self.device == "cuda"
                                else torch.float32
                            ),
                            device_map="auto",
                        )
                    )
                    self.get_model().eval()
                    self.logger.info(
                        f"Pre-trained LLM model '{model_name}' loaded from Transformers."
                    )

            # Move the model to the specified device with the specified CUDA stream
            self.execute_with_stream(lambda: self.model.to(self.device))
            self.logger.info(f"Model moved to {self.device}")

        except Exception as e:
            self.logger.error(f"Error loading model '{model_name}': {e}")

    def set_device(self, device):
        """
        Sets Pytorch device for the model and ensures the model
        is on the correct device.
        """
        self.device = device
        if self.model:
            # Check if the model is already on a valid device and avoid unnecessary transfers
            if "cuda" in device:
                if not torch.cuda.is_available():
                    self.logger.error("CUDA is not available. Falling back to CPU.")
                    self.device = "cpu"
                    self.model = self.model.cpu()
                    return

                try:
                    # Default to cuda:0 if no index provided
                    self.device_index = device.split(":")[-1] if ":" in device else "0"
                    # Set the specific CUDA device
                    torch.cuda.set_device(int(self.device_index))
                    self.execute_with_stream(lambda: self.model.to(self.device))
                    self.logger.info(f"Model moved to device {device}")
                except Exception as e:
                    self.logger.error(f"Failed to set device to {device}: {e}")
                    self.model = self.model.cpu()  # Fallback to CPU if failed
            elif device == "cpu":
                try:
                    # Only move the model if it's not a meta tensor
                    if not any(p.is_meta for p in self.model.parameters()):
                        self.model = self.model.cpu()
                        self.logger.info(f"Model moved to device {device}")
                    else:
                        self.logger.error(
                            "Model contains meta tensors, cannot move to CPU."
                        )
                except Exception as e:
                    self.logger.error(f"Error moving model to CPU: {e}")
            else:
                self.logger.error(f"Invalid device specified: {device}")

    def _forward_classification(self, frame):
        """
        Handles inference for classification models like ResNet.
        Converts tensor output to a list for compatibility.
        """
        self.model.eval()

        # Convert frame to PyTorch tensor
        img_tensor = (
            torch.from_numpy(np.array(frame, copy=True)).permute(2, 0, 1).float()
        )
        img_tensor /= 255.0  # Normalize

        # Move to device
        img_tensor = img_tensor.to(self.device).unsqueeze(0)  # Add batch dimension

        # Run inference
        with torch.inference_mode():
            results = self.model(img_tensor)  # Expected: torch.Tensor

        return results.squeeze().cpu().tolist()  # Convert tensor to list

    def forward(self, frame):
        """Handle inference for different types of models and accumulate frames only for vision-text models."""
        # Initialize frame buffer if it's not set and only for vision-text models

        if self.vision_language_model and self.processor:
            try:
                # Convert the input frame (numpy array) to a PIL Image
                image = Image.fromarray(np.uint8(frame))

                # Define the conversation prompt
                messages = [
                    {
                        "role": "user",
                        "content": f"<|image_1|>\n{self.prompt}",
                    }
                ]

                # Create the input prompt
                prompt = self.processor.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                # Process the input prompt and image for the model
                inputs = self.processor(prompt, [image], return_tensors="pt").to(
                    self.device
                )

                # Set generation parameters
                generation_args = {
                    "max_new_tokens": 500,
                    "temperature": 0.0,
                    "do_sample": False,
                }

                # Generate response
                generate_ids = self.model.generate(
                    **inputs,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    **generation_args,
                )

                # Remove input tokens from the response
                generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]

                # Decode the response
                response = self.processor.batch_decode(
                    generate_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

                self.logger.info(f"Generated response: {response}")

                # Explicitly free memory for unnecessary tensors and clear cache
                del inputs, generate_ids  # Only delete intermediate tensors
                torch.cuda.empty_cache()
                gc.collect()

                return response

            except Exception as e:
                self.logger.error(
                    f"Failed to process frame for Phi-3-vision. Error: {e}"
                )

        elif self.image_processor and self.tokenizer:
            # Add every self.model.config.encoder.num_frames'th
            # incoming frame to the buffer
            self.counter += 1
            if self.counter % self.frame_stride == 0:
                self.frame_buffer.append(frame)

            # Check if we have accumulated enough frames
            if len(self.frame_buffer) >= self.batch_size:
                self.logger.info(f"Processing {self.batch_size} frames")
                try:
                    # generate caption
                    gen_kwargs = {
                        "min_length": 10,
                        "max_length": 20,
                        "num_beams": 8,
                    }
                    pixel_values = self.image_processor(
                        self.frame_buffer, return_tensors="pt"
                    ).pixel_values.to(self.device)
                    tokens = self.model.generate(pixel_values, **gen_kwargs)
                    captions = self.tokenizer.batch_decode(
                        tokens, skip_special_tokens=True
                    )
                    for i, caption in enumerate(captions):
                        self.logger.info(f"{caption}")

                    return captions[0]

                except Exception as e:
                    self.logger.info(f"Failed to process frames. Error: {e}")
                finally:
                    # Clear the frame buffer after processing
                    self.frame_buffer = []

        # Vision-only models:
        elif not self.tokenizer:
            """
            Perform object detection using the PyTorch model.
            """
            # Set the model to evaluation mode to avoid training-related behavior
            self.model.eval()

            # If model is a classification model (like ResNet), return top prediction
            if "resnet" in self.model.__class__.__name__.lower():
                preds = self._forward_classification(frame)  # Run classification

                # ✅ Convert list to NumPy array if necessary
                if isinstance(preds, list):
                    preds = np.array(preds)

                # ✅ Convert Torch tensor to NumPy if necessary
                if isinstance(preds, torch.Tensor):
                    preds = preds.cpu().numpy()

                # ✅ Log predictions
                # self.logger.info(f"Raw predictions shape: {preds.shape}, values: {preds}")

                # Ensure the array has correct dimensions
                if preds.ndim == 1:
                    preds = np.expand_dims(
                        preds, axis=0
                    )  # Convert (num_classes,) → (1, num_classes)

                # ✅ Validate predictions
                if preds.shape[1] == 0:
                    self.logger.error(
                        "Classification model returned empty predictions!"
                    )
                    return None

                # Compute softmax to get probabilities
                probs = np.exp(preds) / np.sum(np.exp(preds), axis=1, keepdims=True)

                # Get the highest probability class
                top_class = np.argmax(probs, axis=1)[0]  # Convert to int
                confidence = float(np.max(probs, axis=1)[0])  # Convert to float

                # ✅ Log classification output
                self.logger.info(
                    f"Top class: {top_class}, Confidence score: {confidence:.4f}"
                )

                # ✅ Check for missing labels/scores
                if top_class is None or confidence is None:
                    self.logger.warning(
                        "Engine: classification result missing label or score."
                    )

                return {
                    "labels": np.array(
                        [top_class]
                    ),  # Ensure labels are in list/array format
                    "scores": np.array(
                        [confidence]
                    ),  # Ensure scores are in list/array format
                }

            # Make a writable copy of the frame to avoid non-writable tensor warnings
            writable_frame = np.array(frame, copy=True)

            # Convert the input frame (likely a NumPy array) to a PyTorch tensor
            img_tensor = torch.from_numpy(writable_frame).permute(2, 0, 1).float()

            # Normalize the tensor if needed
            img_tensor /= 255.0

            # Move the tensor to the appropriate device (e.g., GPU if available)
            img_tensor = img_tensor.to(self.device)

            # Perform inference
            results = None
            if self.device_queue_id is not None and "cuda" in self.device:
                # Create a CUDA stream
                s = torch.cuda.Stream(
                    device=self.device,
                    device_index=self.device_index,
                    device_type=1,  # Use an appropriate value for the device type
                    priority=0,
                    stream_id=self.device_queue_id,
                )
                with torch.cuda.stream(s), torch.inference_mode():
                    results = self.model([img_tensor])[0]
            else:
                with torch.inference_mode():
                    results = self.model([img_tensor])[0]
            # Generic conversion of all PyTorch tensors in the output to NumPy arrays
            output_np = {}
            for key, value in results.items():
                if isinstance(value, torch.Tensor):
                    output_np[key] = (
                        value.cpu().numpy()
                    )  # Convert tensor to NumPy array
                else:
                    output_np[key] = value  # If not a tensor, leave it as is

            return output_np

        # LLM-only models:
        elif self.tokenizer and not self.image_processor:
            try:
                # Assume the frame is a text input for LLM
                inputs = self.tokenizer(frame, return_tensors="pt").to(self.device)

                # Generate text using the LLM model
                with torch.inference_mode():
                    generated_tokens = self.model.generate(**inputs)

                # Decode the generated tokens to text
                generated_text = self.tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )
                self.logger.info(f"Generated text: {generated_text}")

                return generated_text

            except Exception as e:
                self.logger.error(f"Failed to process text input. Error: {e}")

        else:
            raise ValueError("Unsupported model type or missing processor/tokenizer.")

    def generate(self, input_text, max_length=100):
        # Tokenize input text for LLM
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        # Generate text using the model
        outputs = self.model.generate(**inputs, max_length=100)

        # Decode the output to text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.logger.info(f"Generated text: {generated_text}")
        return generated_text

    def execute_with_stream(self, func, *args, **kwargs):
        """
        Execute a function in the context of a CUDA stream if a valid device_queue_id is provided.

        :param func: The function to execute.
        :param args: Positional arguments to pass to the function.
        :param kwargs: Keyword arguments to pass to the function.
        :return: The result of the executed function.
        """
        if self.device_queue_id is not None and "cuda" in self.device:
            # Create a CUDA stream
            s = torch.cuda.Stream(
                device=self.device,
                device_index=self.device_index,
                device_type=1,  # Use an appropriate value for the device type
                priority=0,
                stream_id=self.device_queue_id,
            )
            with torch.cuda.stream(s):
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
