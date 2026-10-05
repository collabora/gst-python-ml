# EngineFactory
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

from importlib import import_module
from typing import Type, Dict

_engine_registry: Dict[str, Type] = {}


class EngineFactory:
    PYTORCH_ENGINE = "pytorch"
    TFLITE_ENGINE = "tflite"
    TENSORFLOW_ENGINE = "tensorflow"
    ONNX_ENGINE = "onnx"
    OPENVINO_ENGINE = "openvino"
    TVM_ENGINE = "tvm"
    TINYGRAD_ENGINE = "tinygrad"
    MLX_ENGINE = "mlx"
    EXECUTORCH_ENGINE = "executorch"
    LLAMACPP_ENGINE = "llamacpp"
    JAX_ENGINE = "jax"
    MIGRAPHX_ENGINE = "migraphx"
    IREE_ENGINE = "iree"
    NCNN_ENGINE = "ncnn"
    DRPAI_ENGINE = "drpai"

    BUILTIN_ENGINES = {
        PYTORCH_ENGINE: ("pytorch_engine", "PyTorchEngine"),
        TFLITE_ENGINE: ("litert_engine", "LiteRTEngine"),
        TENSORFLOW_ENGINE: ("tensorflow_engine", "TensorFlowEngine"),
        ONNX_ENGINE: ("onnx_engine", "ONNXEngine"),
        OPENVINO_ENGINE: ("openvino_engine", "OpenVinoEngine"),
        TVM_ENGINE: ("tvm_engine", "TVMEngine"),
        TINYGRAD_ENGINE: ("tinygrad_engine", "TinyGradEngine"),
        MLX_ENGINE: ("mlx_engine", "MLXEngine"),
        EXECUTORCH_ENGINE: ("executorch_engine", "ExecuTorchEngine"),
        LLAMACPP_ENGINE: ("llamacpp_engine", "LlamaCppEngine"),
        JAX_ENGINE: ("jax_engine", "JAXEngine"),
        MIGRAPHX_ENGINE: ("migraphx_engine", "MiGraphXEngine"),
        IREE_ENGINE: ("iree_engine", "IREEEngine"),
        NCNN_ENGINE: ("ncnn_engine", "NCNNEngine"),
        DRPAI_ENGINE: ("drpai_engine", "DRPAIEngine"),
    }

    @staticmethod
    def register(engine_type: str, engine_class: Type) -> None:
        _engine_registry[engine_type] = engine_class

    @staticmethod
    def create(engine_type: str):
        if not engine_type:
            raise ValueError("Engine type not set")
        if engine_type in _engine_registry:
            return _engine_registry[engine_type]()
        if engine_type not in EngineFactory.BUILTIN_ENGINES:
            raise ValueError(f"Unsupported engine type: {engine_type}")
        module_name, class_name = EngineFactory.BUILTIN_ENGINES[engine_type]
        module = import_module(f".{module_name}", __package__)
        return getattr(module, class_name)()
