# NCNNEngine
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
import ncnn

from .ml_engine import MLEngine


class NCNNEngine(MLEngine):
    def __init__(self):
        super().__init__()
        self.net = None
        self.model_name = None
        self.kwargs = None
        self.input_name = "in0"
        self.output_name = "out0"
        self._use_vulkan = False

    def do_load_model(self, model_name, **kwargs):
        """Load an NCNN model (.param + .bin files).

        model_name should be the path to the .param file.
        The .bin file is expected alongside with the same base name.
        """
        self.model_name = model_name
        self.kwargs = kwargs
        self.input_name = kwargs.get("input_name", "in0")
        self.output_name = kwargs.get("output_name", "out0")

        try:
            # Determine param and bin paths
            if model_name.endswith(".param"):
                param_path = model_name
                bin_path = model_name.replace(".param", ".bin")
            elif model_name.endswith(".bin"):
                bin_path = model_name
                param_path = model_name.replace(".bin", ".param")
            else:
                # Assume base name provided
                param_path = model_name + ".param"
                bin_path = model_name + ".bin"

            if not os.path.isfile(param_path):
                self.logger.error(f"NCNN param file not found: {param_path}")
                return False
            if not os.path.isfile(bin_path):
                self.logger.error(f"NCNN bin file not found: {bin_path}")
                return False

            self.net = ncnn.Net()
            self.net.opt.use_vulkan_compute = self._use_vulkan

            # Set thread count
            num_threads = kwargs.get("num_threads", 4)
            self.net.opt.num_threads = num_threads

            self.net.load_param(param_path)
            self.net.load_model(bin_path)
            self.model = self.net

            self.logger.info(
                f"NCNN model loaded: {param_path} "
                f"(vulkan: {self._use_vulkan}, threads: {num_threads})"
            )
            return True

        except Exception as e:
            self.logger.error(f"Error loading NCNN model '{model_name}': {e}")
            self.net = None
            self.model = None
            return False

    def do_set_device(self, device):
        """Set NCNN compute device."""
        self.device = device

        if "vulkan" in device or "gpu" in device:
            if ncnn.get_gpu_count() > 0:
                self._use_vulkan = True
                self.logger.info(
                    f"NCNN Vulkan GPU enabled ({ncnn.get_gpu_count()} device(s))"
                )
            else:
                self.logger.warning("No Vulkan GPU available, falling back to CPU")
                self._use_vulkan = False
                self.device = "cpu"
        elif device == "cpu":
            self._use_vulkan = False
            self.logger.info("NCNN device set to CPU")
        else:
            self.logger.warning(
                f"Unknown device '{device}' for NCNN, defaulting to CPU"
            )
            self._use_vulkan = False
            self.device = "cpu"

        # Reload model if already loaded
        if self.model_name:
            self.do_load_model(self.model_name, **(self.kwargs or {}))

    def do_forward(self, frames):
        """Run inference using NCNN."""
        if self.net is None:
            self.logger.error("No model loaded")
            return None

        is_batch = isinstance(frames, np.ndarray) and frames.ndim == 4

        # NCNN processes one frame at a time
        if not is_batch:
            frames_list = [frames]
        else:
            frames_list = [frames[i] for i in range(frames.shape[0])]

        results = []
        for frame in frames_list:
            # Convert to float32 and normalize
            img = frame.astype(np.float32) / 255.0

            # NCNN expects CHW format
            if img.ndim == 3 and img.shape[2] in (1, 3, 4):
                img = np.transpose(img, (2, 0, 1))  # HWC -> CHW

            # Create ncnn Mat from numpy
            mat_in = ncnn.Mat(img)

            ex = self.net.create_extractor()
            ex.input(self.input_name, mat_in)
            ret, mat_out = ex.extract(self.output_name)

            if ret != 0:
                self.logger.error(f"NCNN extract failed with code {ret}")
                results.append(None)
                continue

            # Convert ncnn Mat to numpy
            out = np.array(mat_out)
            results.append(out)

        if not is_batch:
            # ncnn drops the batch axis the decoder expects
            raw = results[0][None] if results[0] is not None else None
        else:
            raw = np.stack(results) if all(r is not None for r in results) else results

        return self._apply_post_process(raw, is_batch)

    def do_generate(self, input_text, max_length=1000, system_prompt=None):
        """NCNN does not support text generation."""
        raise NotImplementedError(
            "NCNN is a vision inference framework and does not support "
            "text generation. Use PyTorch or llama.cpp for LLM workloads."
        )
