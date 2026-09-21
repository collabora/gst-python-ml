# PyTorchEngine
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

import os
import numpy as np
import traceback

from .ml_engine import MLEngine


# transformers 5 returns a model output from get_*_features, 4 returned the tensor
def projected(features):
    return getattr(features, "pooler_output", features)


class PyTorchEngine(MLEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_executorch = False
        self.is_tensorrt = False  # Flag to track if TensorRT is used

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
        try:
            import torch_tensorrt
        except ImportError:
            torch_tensorrt = None

        processor_name = kwargs.get("processor_name")
        tokenizer_name = kwargs.get("tokenizer_name")
        compile_model = kwargs.get("compile", False)
        use_tensorrt = kwargs.get("use_tensorrt", False)  # Kwarg to enable TensorRT
        trt_precision = kwargs.get(
            "trt_precision", "fp16"
        )  # Optional: fp16, int8, etc.
        trt_input_shapes = kwargs.get(
            "trt_input_shapes", None
        )  # User-provided shapes, e.g., [(batch, channels, height, width)]

        try:
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
                    self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
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
                    self.execute_with_stream(lambda: self.model.to(self.device))
                    self.logger.info(f"Model moved to {self.device}")

                # Compile with TensorRT if enabled and available
                if (
                    use_tensorrt
                    and torch_tensorrt is not None
                    and "cuda" in self.device
                ):
                    if not torch.cuda.is_available():
                        self.logger.warning("TensorRT requires CUDA; skipping.")
                    else:
                        # Require input shapes; fail if not provided
                        if trt_input_shapes is None:
                            raise ValueError(
                                "trt_input_shapes must be provided when using TensorRT. "
                                "Example: [(1, 3, 224, 224)] for fixed shape or dict for dynamic."
                            )

                        # Convert shapes to torch_tensorrt.Input objects (supports fixed or dynamic)
                        trt_inputs = []
                        for shape in trt_input_shapes:
                            if isinstance(shape, tuple):  # Fixed shape
                                trt_inputs.append(torch_tensorrt.Input(shape))
                            elif isinstance(
                                shape, dict
                            ):  # Dynamic: {'min': (1,3,224,224), 'opt': ..., 'max': ...}
                                trt_inputs.append(torch_tensorrt.Input(**shape))
                            else:
                                raise ValueError(
                                    f"Invalid trt_input_shape format: {shape}"
                                )

                        try:
                            self.model = torch_tensorrt.compile(
                                self.model,
                                inputs=trt_inputs,
                                enabled_precisions={
                                    (
                                        torch.half
                                        if trt_precision == "fp16"
                                        else torch.float
                                    )
                                },  # FP16 for speed
                                workspace_size=1
                                << 32,  # 4GB workspace; adjust as needed
                            )
                            self.is_tensorrt = True
                            self.logger.info(
                                f"Model compiled with TensorRT ({trt_precision} precision) using provided shapes"
                            )
                        except Exception as e:
                            self.logger.error(f"Failed to compile with TensorRT: {e}")
                            self.logger.warning("Falling back to standard PyTorch")
                elif use_tensorrt:
                    self.logger.warning(
                        "torch-tensorrt not installed; install with 'pip install torch-tensorrt'"
                    )

                if (
                    compile_model and not self.is_tensorrt
                ):  # Standard torch.compile if not using TRT
                    self.model = torch.compile(self.model)
                    self.logger.info("Model compiled with torch.compile")

            return True

        except Exception as e:
            stack_trace = (
                traceback.format_stack()
            )  # Capture the current call stack as a list of strings
            self.logger.error(
                f"Error loading model '{model_name}': {e}"
                f"Stack trace:\n{''.join(stack_trace)}"  # Join and log the stack trace
            )
            self.tokenizer = None
            self.model = None
            return False

    def do_set_device(self, device):
        """Set PyTorch device for the model (ExecuTorch handles devices via export/backends)."""
        import torch

        self.device = device
        self.logger.info(f"Setting device to {device}")

        if "cuda" in device:
            if not torch.cuda.is_available():
                self.logger.warning("CUDA is not available, falling back to CPU")
                self.device = "cpu"
                if self.model and not self.is_executorch and hasattr(self.model, "to"):
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

                # Model placement is handled by device_map="auto" in do_load_model
                # Only move model if it exists and is not already on the correct device
                if self.model and not self.is_executorch and hasattr(self.model, "to"):
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
                if self.model and not self.is_executorch and hasattr(self.model, "to"):
                    self.model = self.model.cpu()

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
            self.logger.error(f"Invalid device specified: {device}")
            self.device = "cpu"
            if self.model and not self.is_executorch and hasattr(self.model, "to"):
                self.model = self.model.cpu()

    def _forward_classification(self, frames):
        """Handle inference for classification models like ResNet (non-ExecuTorch only)."""
        import torch

        if self.is_executorch:
            raise NotImplementedError("Classification not adapted for ExecuTorch yet")
        self.model.eval()
        is_batch = frames.ndim == 4  # (B, H, W, C) vs (H, W, C)
        # Create tensor and normalize (moved here for consistency)
        img_tensor = torch.from_numpy(np.array(frames, copy=True)).float() / 255.0

        if is_batch:
            img_tensor = img_tensor.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        else:
            img_tensor = img_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            img_tensor = img_tensor.unsqueeze(0)  # Add batch dim: (1, C, H, W)

        img_tensor = img_tensor.to(self.device)

        with torch.inference_mode():
            results = self.model(img_tensor)
        return (
            results.squeeze(0) if not is_batch else results
        )  # Squeeze batch dim if single

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
            # torch.compile wraps the model in OptimizedModule; unwrap for name check
            inner = getattr(self.model, "_orig_mod", self.model)
            if "resnet" in inner.__class__.__name__.lower():
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
                results = self.model(
                    img_tensor
                )  # Batch inference (works with TRT-compiled model)

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

    def do_generate(self, input_text, max_length=1000, system_prompt=None):
        import torch

        try:
            import executorch.pybindings as et
        except ImportError:
            et = None
        messages = [{"role": "user", "content": input_text}]
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # Switches between thinking and non-thinking modes. Default is True.
        )

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
        outputs = self.model.generate(**inputs, max_length=max_length)
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

    def execute_with_stream(self, func, *args, **kwargs):
        import torch

        if self.device_queue_id is not None and "cuda" in self.device:
            s = torch.cuda.Stream(
                device=self.device, priority=0, stream_id=self.device_queue_id
            )
            with torch.cuda.stream(s):
                return func(*args, **kwargs)
        return func(*args, **kwargs)
