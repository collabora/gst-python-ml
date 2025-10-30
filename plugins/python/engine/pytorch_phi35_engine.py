# PyTorchPhi35Engine
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


import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
)
import numpy as np
from .pytorch_engine import PyTorchEngine


class PyTorchPhi35Engine(PyTorchEngine):
    def load_model(self, model_name, **kwargs):
        """Load a Phi-3-vision model from Hugging Face."""
        try:
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
            self.logger.info("Phi-3.5-vision model and processor loaded successfully.")
            self.model.eval()

            # Skip .to() for 4-bit models
            if not (
                hasattr(self.model, "is_loaded_in_4bit")
                and self.model.is_loaded_in_4bit
            ):
                self.execute_with_stream(lambda: self.model.to(self.device))
                self.logger.info(f"Model moved to {self.device}")

            return True

        except Exception as e:
            self.logger.error(f"Error loading model '{model_name}': {e}")
            self.tokenizer = None
            self.model = None
            return False

    def forward(self, frames):
        """Handle inference for phi3.5 vision, supporting single frames or batches."""
        is_batch = isinstance(frames, np.ndarray) and frames.ndim == 4  # (B, H, W, C)
        if not isinstance(frames, (np.ndarray, str)):
            self.logger.error(f"Invalid input type for forward: {type(frames)}")
            return None

        if self.processor:
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
