# PyTorchVisionEngine
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from abc import abstractmethod
import numpy as np
import gc

from engine.pytorch_engine import PyTorchEngine


class PyTorchVisionEngine(PyTorchEngine):
    def do_forward(self, frames):
        import torch

        is_batch = isinstance(frames, np.ndarray) and frames.ndim == 4
        if not isinstance(frames, (np.ndarray, str)):
            self.logger.error(f"Invalid input type for forward: {type(frames)}")
            return None

        from PIL import Image

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
