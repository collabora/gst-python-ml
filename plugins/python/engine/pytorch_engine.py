# PyTorchEngine
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import functools
import os
import numpy as np

from .hub_causal_lm import chat_prompt
from .ml_engine import MLEngine


# transformers 5 returns a model output from get_*_features, 4 returned the tensor
def projected(features):
    return getattr(features, "pooler_output", features)


INFERENCE_METHOD_NAMES = ("do_forward", "do_generate")
DEFAULT_STREAM_PRIORITY = 0


def on_the_engine_stream(method):
    @functools.wraps(method)
    def run_on_the_engine_stream(self, *args, **kwargs):
        with self.stream_context():
            return method(self, *args, **kwargs)

    return run_on_the_engine_stream


class PyTorchEngine(MLEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_executorch = False
        self._stream_priority = DEFAULT_STREAM_PRIORITY
        self._stream = None

    # a subclass's inference runs on the engine's stream without asking
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name in INFERENCE_METHOD_NAMES:
            if name in cls.__dict__:
                setattr(cls, name, on_the_engine_stream(cls.__dict__[name]))

    @property
    def stream_priority(self):
        return self._stream_priority

    # a cuda stream keeps the priority it was created with
    @stream_priority.setter
    def stream_priority(self, priority):
        self._stream_priority = priority
        self._stream = None

    # created on first use, a test may name cuda on a box without it
    @property
    def stream(self):
        if self._stream is None:
            self._stream = self.new_cuda_stream()
        return self._stream

    def new_cuda_stream(self):
        if self.device is None or "cuda" not in self.device:
            return None
        import torch

        return torch.cuda.Stream(
            device=int(self.device_index), priority=self._stream_priority
        )

    def stream_context(self):
        if self.stream is None:
            return super().stream_context()
        import torch

        return torch.cuda.stream(self.stream)

    def do_load_model(self, model_name, **kwargs):
        """Load a pre-trained model by name from TorchVision, Transformers, or a local path (including .pte for ExecuTorch)."""
        import torch
        from torchvision import models
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            AutoImageProcessor,
            VisionEncoderDecoderModel,
            BitsAndBytesConfig,
        )

        try:
            import executorch.pybindings as et
        except ImportError:
            et = None

        processor_name = kwargs.get("processor_name")
        tokenizer_name = kwargs.get("tokenizer_name")
        compile_model = kwargs.get("compile", False)

        if os.path.isfile(model_name):
            if model_name.endswith(".pte") and et is not None:
                # Load ExecuTorch model
                self.model = et.Module(model_name)  # Load .pte file
                self.is_executorch = True
                self.logger.info(f"ExecuTorch model loaded from: {model_name}")
            else:
                self.model = torch.load(model_name)
                self.logger.info(f"Model loaded from local path: {model_name}")
        else:
            if hasattr(models.detection, model_name):
                weights = models.get_model_weights(model_name).DEFAULT
                self.model = getattr(models.detection, model_name)(weights=weights)
                self.categories = weights.meta["categories"]
                self.logger.info(
                    f"Pre-trained detection model '{model_name}' loaded from TorchVision.detection"
                )
            elif processor_name and tokenizer_name:
                self.image_processor = AutoImageProcessor.from_pretrained(
                    processor_name
                )
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
                self.frame_stride = self.model.config.encoder.num_frames
                self.logger.info(
                    f"Vision-Text model '{model_name}' loaded with processor and tokenizer."
                )
            else:
                self.logger.info(f"Loading tokenizer for language model {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.logger.info(f"Loading language model {model_name}")
                quantization_config = BitsAndBytesConfig(load_in_4bit=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=(
                        torch.float16 if self.device == "cuda" else torch.float32
                    ),
                    device_map="auto",
                    quantization_config=quantization_config,
                )

                self.get_model().eval()
                self.logger.info(
                    f"Pre-trained LLM model '{model_name}' loaded from Transformers."
                )

        if not self.is_executorch:
            if hasattr(self.model, "to") and callable(getattr(self.model, "to")):
                self.model.to(self.device)
                self.logger.info(f"Model moved to {self.device}")

            if compile_model:
                self.model = torch.compile(self.model)
                self.logger.info("Model compiled with torch.compile")

        return True

    def do_set_device(self, device):
        import torch

        if "cuda" in device:
            if not torch.cuda.is_available():
                raise RuntimeError(f"torch sees no CUDA device for device={device}")
            # Extract device index (e.g., "cuda:0" -> "0")
            self.device_index = device.split(":")[-1] if ":" in device else "0"
            torch.cuda.set_device(int(self.device_index))
            self.logger.info(f"CUDA device set to cuda:{self.device_index}")

            # Model placement is handled by device_map="auto" in do_load_model
            # Only move model if it exists and is not already on the correct device
            if self.model and not self.is_executorch and hasattr(self.model, "to"):
                current_devices = {param.device for param in self.model.parameters()}
                if any(d.type != "cuda" for d in current_devices):
                    self.logger.warning(
                        f"Model tensors found on {current_devices}, moving to {device}"
                    )
                    self.model = self.model.to(device)
                    self.logger.info(f"Model moved to {device}")

        elif device == "cpu":
            if self.model and not self.is_executorch and hasattr(self.model, "to"):
                try:
                    if not any(p.is_meta for p in self.model.parameters()):
                        self.model = self.model.cpu()
                        self.logger.info("Model moved to CPU")
                    else:
                        self.logger.error(
                            "Model contains meta tensors, cannot move to CPU"
                        )
                except Exception as e:
                    self.logger.error(f"Error moving model to CPU: {e}")

        else:
            raise ValueError(f"Invalid device specified: {device}")
        self.device = device
        self._stream = None
        self.logger.info(f"Setting device to {device}")

    @on_the_engine_stream
    def do_forward(self, frames):
        """Handle inference for different types of models, supporting single frames or batches (with ExecuTorch adaptation)."""
        import torch

        try:
            import executorch.pybindings as et
        except ImportError:
            et = None
        if self.is_executorch:
            if not isinstance(frames, np.ndarray):
                self.logger.error(
                    f"Invalid input for ExecuTorch forward: {type(frames)}"
                )
                return None
            # Convert NumPy to ExecuTorch tensor (adapt as needed for your models)
            et_tensor = et.Tensor(frames)  # Assuming ExecuTorch tensor from NumPy
            outputs = self.model.forward(et_tensor)  # ExecuTorch forward
            # Convert back to NumPy or your expected format
            if isinstance(outputs, et.Tensor):
                outputs = outputs.numpy()
            self.logger.info(f"ExecuTorch forward results: {outputs}")
            return outputs  # Adapt for batch/single as needed

        is_batch = isinstance(frames, np.ndarray) and frames.ndim == 4  # (B, H, W, C)
        if not isinstance(frames, (np.ndarray, str)):
            self.logger.error(f"Invalid input type for forward: {type(frames)}")
            return None

        if self.image_processor and self.tokenizer:
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
                gen_kwargs = {"min_length": 10, "max_length": 20, "num_beams": 8}
                pixel_values = self.image_processor(
                    self.frame_buffer, return_tensors="pt"
                ).pixel_values.to(self.device)
                tokens = self.model.generate(pixel_values, **gen_kwargs)
                captions = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
                self.logger.info(f"Captions: {captions}")
                self.frame_buffer = []
                return captions[0]
            return None

        elif not self.tokenizer:
            self.model.eval()
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
                results = self.model(img_tensor)

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

    @on_the_engine_stream
    def do_generate(self, input_text, max_length=1000, system_prompt=None):
        import torch

        try:
            import executorch.pybindings as et
        except ImportError:
            et = None
        input_text = chat_prompt(self.tokenizer, input_text, system_prompt)

        if self.is_executorch:
            if not self.tokenizer:
                raise ValueError("Tokenizer required for ExecuTorch generation")
            # Adapt for ExecuTorch (e.g., for LLMs; use tokenizer as usual)
            inputs = self.tokenizer(input_text, return_tensors="pt")
            et_inputs = et.Tensor(
                inputs["input_ids"].numpy()
            )  # Convert to ExecuTorch tensor
            outputs = self.model.forward(et_inputs)  # Or use generate if supported
            generated_tokens = torch.from_numpy(
                outputs.numpy()
            )  # Back to torch for decode
            generated_text = self.tokenizer.decode(
                generated_tokens[0], skip_special_tokens=True
            )
            self.logger.info(f"ExecuTorch generated text: {generated_text}")
            return generated_text

        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_length)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        outputs = outputs[0][len(inputs.input_ids[0]) :].tolist()
        try:
            # rindex finding 151668 (</think>)
            index = len(outputs) - outputs[::-1].index(151668)
        except ValueError:
            index = 0
        generated_text = self.tokenizer.decode(
            outputs[index:], skip_special_tokens=True
        )

        self.logger.info(f"Generated text: {generated_text}")
        return generated_text
