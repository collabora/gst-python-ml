# OpenVinoEngine
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
import openvino as ov

from .ml_engine import MLEngine, TORCHVISION_WEIGHTS


class OpenVinoEngine(MLEngine):
    def __init__(self):
        super().__init__()
        self.core = ov.Core()
        self.compiled_model = None
        self.ov_model = None
        self.model_type = None
        self.model_name = None
        self.kwargs = None
        self.detection_threshold = 0.5

    def do_load_model(self, model_name, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

        if model_name.endswith((".xml", ".bin")):
            base = model_name.rsplit(".", 1)[0]
        else:
            base = model_name
        xml_path = f"{base}.xml"
        bin_path = f"{base}.bin"
        if os.path.isfile(xml_path) and os.path.isfile(bin_path):
            self.ov_model = self.core.read_model(xml_path)
            self.model_type = "custom"
            self.logger.info(f"OpenVINO IR model loaded from local path: {model_name}")
        else:
            from torchvision import models

            if hasattr(models, model_name):
                pt_model = getattr(models, model_name)(weights=TORCHVISION_WEIGHTS)
                self.ov_model = ov.convert_model(pt_model)
                self.model_type = "classification"
                self.logger.info(
                    f"Pre-trained vision model '{model_name}' converted to OpenVINO."
                )
            elif hasattr(models.detection, model_name):
                pt_model = getattr(models.detection, model_name)(
                    weights=TORCHVISION_WEIGHTS
                )
                self.ov_model = ov.convert_model(pt_model)
                self.model_type = "detection"
                self.logger.info(
                    f"Pre-trained detection model '{model_name}' converted to OpenVINO."
                )
            else:
                raise FileNotFoundError(
                    "OpenVINO takes an IR .xml with its .bin or a torchvision model name, "
                    f"got: {model_name}"
                )

        self.compiled_model = self.core.compile_model(self.ov_model, self.device)
        self.logger.info(f"Model compiled on {self.device}")

        return True

    def do_set_device(self, device):
        """Set OpenVINO device for the model."""
        # Map cuda/gpu aliases to OpenVINO device names (uppercase)
        if "cuda" in device.lower() or device.lower() == "gpu":
            self.device = "GPU"
        else:
            self.device = device.upper()
        self.logger.info(f"Setting device to {self.device}")

        available_devices = self.core.available_devices
        if self.device not in available_devices:
            if "GPU" in self.device and any("GPU" in d for d in available_devices):
                self.device = "GPU"
            else:
                raise RuntimeError(
                    f"OpenVINO has no device {self.device}, "
                    f"it has {', '.join(available_devices)}"
                )

        # Recompile model if already loaded
        if self.model_name:
            self.do_load_model(self.model_name, **self.kwargs)

    def _forward_classification(self, frames):
        """Handle inference for classification models."""
        is_batch = frames.ndim == 4
        img_array = np.array(frames, dtype=np.float32) / 255.0
        if is_batch:
            img_array = np.transpose(
                img_array, (0, 3, 1, 2)
            )  # (B, H, W, C) -> (B, C, H, W)
        else:
            img_array = np.transpose(img_array, (2, 0, 1))  # (H, W, C) -> (C, H, W)
            img_array = np.expand_dims(img_array, 0)

        infer_request = self.compiled_model.create_infer_request()
        infer_request.infer({0: img_array})
        preds = infer_request.get_output_tensor(0).data
        return self._top_classes(preds, is_batch)

    def do_forward(self, frames):
        """Handle inference for different types of models, supporting single frames or batches."""
        is_batch = isinstance(frames, np.ndarray) and frames.ndim == 4
        if not isinstance(frames, np.ndarray):
            self.logger.error(f"Invalid input type for forward: {type(frames)}")
            return None

        if self.model_type == "classification":
            return self._forward_classification(frames)

        elif self.model_type == "detection":
            writable_frames = np.array(frames, copy=True, dtype=np.float32) / 255.0
            if is_batch:
                img_array = np.transpose(
                    writable_frames, (0, 3, 1, 2)
                )  # (B, H, W, C) -> (B, C, H, W)
            else:
                img_array = np.transpose(writable_frames, (2, 0, 1))
                img_array = np.expand_dims(img_array, 0)
            infer_request = self.compiled_model.create_infer_request()
            infer_request.infer({0: img_array})
            outputs = [
                infer_request.get_output_tensor(i).data
                for i in range(len(self.compiled_model.outputs))
            ]
            # Assuming outputs for detection: [boxes, labels, scores]
            if len(outputs) == 3:
                boxes, labels, scores = outputs
                results = []
                for i in range(img_array.shape[0]):
                    valid = scores[i] > self.detection_threshold
                    res = {
                        "boxes": boxes[i][valid],
                        "labels": labels[i][valid].astype(int),
                        "scores": scores[i][valid],
                    }
                    results.append(res)
            else:
                raise ValueError("Unexpected output format for detection model.")
            self.logger.debug(
                f"Batch inference results: {len(results)} frames processed"
            )
            return results[0] if not is_batch else results

        elif self.model_type == "custom":
            # Generic forward for custom models
            img = self._apply_input_format(frames.astype(np.float32) / 255.0, is_batch)
            infer_request = self.compiled_model.create_infer_request()
            infer_request.infer({0: img})
            outputs = [
                infer_request.get_output_tensor(i).data
                for i in range(len(self.compiled_model.outputs))
            ]
            raw = outputs if len(outputs) > 1 else outputs[0]
            return self._apply_post_process(raw, is_batch)

        else:
            raise ValueError("Unsupported model type.")

    def do_generate(self, input_text, max_length=1000, system_prompt=None):
        raise NotImplementedError(
            "OpenVINO does not support text generation. "
            "Use PyTorch or llama.cpp for LLM workloads."
        )
