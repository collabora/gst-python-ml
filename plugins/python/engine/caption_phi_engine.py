# CaptionPhiEngine
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Library General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
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

from .pytorch_vision_engine import PyTorchVisionEngine


class CaptionPhiEngine(PyTorchVisionEngine):
    def do_load_model(self, model_name, **kwargs):
        """Load a Phi-3-vision model from Hugging Face."""
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            _attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.logger.info("Phi-3.5-vision model and processor loaded successfully.")
        self.model.eval()

        # Skip .to() for 4-bit models
        if not (
            hasattr(self.model, "is_loaded_in_4bit") and self.model.is_loaded_in_4bit
        ):
            self.execute_with_stream(lambda: self.model.to(self.device))
            self.logger.info(f"Model moved to {self.device}")

        return True

    def _prepare_messages(self, images):
        prompt_content = (
            "\n".join([f"<|image_{i+1}|>" for i in range(len(images))])
            + f"\n{self.prompt}"
        )
        return [{"role": "user", "content": prompt_content}]

    def _process_inputs(self, prompt_text, images):
        return self.processor(prompt_text, images, return_tensors="pt").to(self.device)

    def _trim_generated_ids(self, inputs, generate_ids):
        return generate_ids[:, inputs["input_ids"].shape[1] :]
