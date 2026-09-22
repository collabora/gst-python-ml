# VlmEngine
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

from .pytorch_engine import PyTorchEngine


class VlmEngine(PyTorchEngine):
    """
    PyTorch engine for Vision-Language Models.

    Supports HuggingFace VLM model IDs via AutoProcessor + AutoModelForVision2Seq:
      llava-hf/llava-1.5-7b-hf
      Qwen/Qwen2-VL-7B-Instruct
      OpenGVLab/InternVL2-8B
    """

    def __init__(self):
        super().__init__()
        self.processor = None

    def do_load_model(self, model_name, **kwargs):
        try:
            import torch
            import transformers
            from transformers import AutoProcessor

            # transformers 5 renamed AutoModelForVision2Seq
            loader = getattr(
                transformers,
                "AutoModelForImageTextToText",
                getattr(transformers, "AutoModelForVision2Seq", None),
            )
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = loader.from_pretrained(
                model_name,
                dtype=torch.float16,
                device_map=self.device,
            )
            self.model.eval()
            self.logger.info(f"VLM model '{model_name}' loaded on {self.device}")
        except Exception as e:
            raise ValueError(f"Failed to load VLM model '{model_name}': {e}")

    def do_forward(
        self,
        frame,
        prompt="Describe this image in detail.",
        system_prompt=None,
        max_tokens=256,
        temperature=0.7,
    ):
        """
        Run VLM inference on a single video frame.

        Args:
            frame: numpy RGB array (H, W, 3).
            prompt: user prompt text.
            system_prompt: optional system prompt.
            max_tokens: maximum tokens to generate.
            temperature: sampling temperature.

        Returns:
            Generated text string, or None on failure.
        """
        import numpy as np
        from PIL import Image

        pil_img = Image.fromarray(frame.astype(np.uint8))
        text = self.do_generate(pil_img, prompt, system_prompt, max_tokens, temperature)
        return text

    def do_generate(self, image, prompt, system_prompt, max_tokens, temperature):
        """
        Apply chat template, process image + text, and generate a response.

        Args:
            image: PIL Image.
            prompt: user prompt text.
            system_prompt: optional system prompt.
            max_tokens: maximum new tokens.
            temperature: sampling temperature.

        Returns:
            Generated text string.
        """
        import torch

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        )

        # Apply chat template if the processor supports it
        if hasattr(self.processor, "apply_chat_template"):
            text_input = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
        else:
            text_input = prompt

        inputs = self.processor(text=text_input, images=image, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )

        # Decode only the newly generated tokens
        input_len = inputs.get("input_ids", torch.tensor([])).shape[-1]
        generated = output_ids[0][input_len:]
        text = self.processor.decode(generated, skip_special_tokens=True)
        return text.strip()
