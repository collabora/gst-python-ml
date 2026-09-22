# PyTorchVisionEngine
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

from abc import abstractmethod
import numpy as np
from PIL import Image
import gc

from engine.pytorch_engine import PyTorchEngine


class PyTorchVisionEngine(PyTorchEngine):
    def do_forward(self, frames):
        import torch

        is_batch = isinstance(frames, np.ndarray) and frames.ndim == 4
        if not isinstance(frames, (np.ndarray, str)):
            self.logger.error(f"Invalid input type for forward: {type(frames)}")
            return None

        # Shared: Convert to PIL
        images = (
            [Image.fromarray(np.uint8(frame)) for frame in frames]
            if is_batch
            else [Image.fromarray(np.uint8(frames))]
        )

        # Model-specific: Prepare messages and prompt
        messages = self._prepare_messages(images)
        prompt_text = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Model-specific: Process inputs
        inputs = self._process_inputs(prompt_text, images)

        # Shared: Inference
        generation_args = {
            "max_new_tokens": 100,
            "temperature": 0.0,
            "do_sample": False,
        }
        with torch.inference_mode():
            generate_ids = self.model.generate(
                **inputs,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                **generation_args,
            )

        # Shared: Trim and decode (adapt for model differences)
        generate_ids_trimmed = self._trim_generated_ids(inputs, generate_ids)
        response = self.processor.batch_decode(
            generate_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Shared: Split for batch
        if is_batch:
            captions = (
                response.split("\n")[: len(images)]
                if "\n" in response
                else [response] * len(images)
            )
        else:
            captions = [response]

        self.logger.info(f"Generated captions: {captions}")

        # Shared: Cleanup
        del inputs, generate_ids
        torch.cuda.empty_cache()
        gc.collect()

        return captions if is_batch else captions[0]

    # Abstract methods for model-specific parts
    @abstractmethod
    def _prepare_messages(self, images):
        pass

    @abstractmethod
    def _process_inputs(self, prompt_text, images):
        pass

    @abstractmethod
    def _trim_generated_ids(self, inputs, generate_ids):
        pass
