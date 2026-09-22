# CaptionQwenEngine
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


class CaptionQwenEngine(PyTorchVisionEngine):
    def do_load_model(self, model_name, **kwargs):
        """Load a Qwen2.5-VL model from Hugging Face."""
        import torch
        from transformers import (
            AutoConfig,
            AutoProcessor,
            Qwen2_5_VLForConditionalGeneration,
        )

        config = AutoConfig.from_pretrained(model_name)
        quantization = getattr(config, "quantization_config", None)
        if isinstance(quantization, dict):
            # awq hub configs skip "visual", transformers 5 names it "model.visual"
            quantization["modules_to_not_convert"] = [
                "model.visual" if module == "visual" else module
                for module in quantization.get("modules_to_not_convert") or []
            ]
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            config=config,
            dtype=torch.float16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

        self.logger.info(f"{model_name} model and processor loaded successfully.")
        self.model.eval()

        # Skip .to() for quantized models
        if not (hasattr(self.model, "is_quantized") and self.model.is_quantized):
            self.execute_with_stream(lambda: self.model.to(self.device))
            self.logger.info(f"Model moved to {self.device}")

    def _prepare_messages(self, images):
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": self.prompt})
        return [{"role": "user", "content": content}]

    def _process_inputs(self, prompt_text, images):
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(
            self._prepare_messages(images)
        )  # Note: Uses messages directly
        return self.processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

    def _trim_generated_ids(self, inputs, generate_ids):
        return [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generate_ids)
        ]
