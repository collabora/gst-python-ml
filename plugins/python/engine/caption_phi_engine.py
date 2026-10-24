# CaptionPhiEngine
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

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
            self.model.to(self.device)
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
