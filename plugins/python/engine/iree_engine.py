# IREEEngine
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
import subprocess
import tempfile
import numpy as np
from iree import runtime as ireert

from .ml_engine import MLEngine

# iree-import-onnx names the graph entry point main_graph
ONNX_IMPORT_FUNCTION = "main_graph"


class IREEEngine(MLEngine):
    def __init__(self):
        super().__init__()
        self.module = None
        self.config = None
        self.context = None
        self.function_name = None
        self.model_name = None
        self.kwargs = None
        self._driver = "local-task"

    def _driver_for_device(self, device):
        """Map device string to IREE driver name."""
        if device in ("cpu", "local-task"):
            return "local-task"
        elif device in ("local-sync",):
            return "local-sync"
        elif "rocm" in device or "hip" in device or "gpu" in device:
            return "hip"
        elif "vulkan" in device:
            return "vulkan"
        elif "cuda" in device:
            return "cuda"
        elif "metal" in device:
            return "metal"
        return "local-task"

    def _target_backend_for_driver(self, driver):
        """Map IREE driver to compilation target backend."""
        mapping = {
            "local-task": "llvm-cpu",
            "local-sync": "llvm-cpu",
            "hip": "rocm",
            "vulkan": "vulkan-spirv",
            "cuda": "cuda",
            "metal": "metal",
        }
        return mapping.get(driver, "llvm-cpu")

    def _compile_onnx(self, onnx_path):
        """Compile an ONNX model to IREE vmfb format."""
        from iree import compiler as ireec

        target_backend = self._target_backend_for_driver(self._driver)

        # Step 1: Import ONNX to MLIR using iree-import-onnx
        with tempfile.NamedTemporaryFile(suffix=".mlir", delete=False) as mlir_file:
            mlir_path = mlir_file.name

        try:
            result = subprocess.run(
                [
                    "iree-import-onnx",
                    onnx_path,
                    "--opset-version",
                    "17",
                    "-o",
                    mlir_path,
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                self.logger.error(f"iree-import-onnx failed: {result.stderr}")
                return None

            # Step 2: Compile MLIR to vmfb
            compiled = ireec.tools.compile_file(
                mlir_path,
                target_backends=[target_backend],
            )
            return compiled
        finally:
            if os.path.exists(mlir_path):
                os.unlink(mlir_path)

    def do_load_model(self, model_name, **kwargs):
        """Load a pre-compiled .vmfb module or compile from .onnx."""
        self.model_name = model_name
        self.kwargs = kwargs
        default_function = (
            ONNX_IMPORT_FUNCTION if model_name.endswith(".onnx") else "forward"
        )
        self.function_name = kwargs.get("function_name", default_function)

        try:
            if model_name.endswith(".vmfb"):
                # Load pre-compiled module
                if not os.path.isfile(model_name):
                    self.logger.error(f"VMFB file not found: {model_name}")
                    return False

                self.config = ireert.Config(self._driver)
                self.context = ireert.SystemContext(config=self.config)
                with open(model_name, "rb") as f:
                    vmfb_data = f.read()
                vm_module = ireert.VmModule.copy_buffer(
                    self.context.instance, vmfb_data
                )
                self.context.add_vm_module(vm_module)
                self.model = self.context
                self.logger.info(
                    f"IREE module loaded from {model_name} (driver: {self._driver})"
                )
                return True

            elif model_name.endswith(".onnx"):
                # Compile from ONNX
                if not os.path.isfile(model_name):
                    self.logger.error(f"ONNX file not found: {model_name}")
                    return False

                self.logger.info(
                    f"Compiling ONNX model {model_name} for {self._driver}..."
                )
                compiled = self._compile_onnx(model_name)
                if compiled is None:
                    return False

                self.config = ireert.Config(self._driver)
                self.context = ireert.SystemContext(config=self.config)
                vm_module = ireert.VmModule.copy_buffer(self.context.instance, compiled)
                self.context.add_vm_module(vm_module)
                self.model = self.context
                self.logger.info(
                    f"IREE model compiled and loaded from {model_name} "
                    f"(driver: {self._driver})"
                )
                return True
            else:
                self.logger.error(
                    f"IREE requires a .vmfb or .onnx file, got: {model_name}"
                )
                return False

        except Exception as e:
            self.logger.error(f"Error loading IREE model '{model_name}': {e}")
            self.model = None
            self.context = None
            return False

    def do_set_device(self, device):
        """Set the IREE runtime driver/device."""
        self.device = device
        self._driver = self._driver_for_device(device)
        self.logger.info(f"IREE driver set to {self._driver}")

        # Reload model if already loaded
        if self.model_name:
            self.do_load_model(self.model_name, **(self.kwargs or {}))

    def do_forward(self, frames):
        """Run inference using the IREE runtime."""
        if self.context is None:
            self.logger.error("No model loaded")
            return None

        is_batch = isinstance(frames, np.ndarray) and frames.ndim == 4

        # Apply input format
        fmt = self.input_format
        if fmt == "auto":
            self.input_format = "nchw"
        img = self._apply_input_format(frames.astype(np.float32) / 255.0, is_batch)

        # Find the module and function
        try:
            # IREE modules are accessible by name; typically "module" for ONNX imports
            module_name = (
                self.kwargs.get("module_name", "module") if self.kwargs else "module"
            )
            f = self.context.modules[module_name][self.function_name]
            result = f(img)
            raw = (
                np.asarray(result.to_host())
                if hasattr(result, "to_host")
                else np.asarray(result)
            )
        except Exception as e:
            self.logger.error(f"IREE inference failed: {e}")
            return None

        return self._apply_post_process(raw, is_batch)

    def do_generate(self, input_text, max_length=1000, system_prompt=None):
        """IREE does not natively support text generation."""
        raise NotImplementedError(
            "IREE is a compiled-model inference runtime and does not support "
            "text generation directly. Use PyTorch or llama.cpp for LLM workloads."
        )
