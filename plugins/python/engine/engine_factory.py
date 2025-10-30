# EngineFactory
# Copyright (C) 2024-2025 Collabora Ltd.
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

from typing import Type, Dict

_engine_registry: Dict[str, Type] = {}


def _try_register(name: str, cls: Type) -> None:
    try:
        _engine_registry[name] = cls
    except Exception:
        pass


class EngineFactory:
    PYTORCH_ENGINE = "pytorch"
    TFLITE_ENGINE = "tflite"
    TENSORFLOW_ENGINE = "tensorflow"
    ONNX_ENGINE = "onnx"
    OPENVINO_ENGINE = "openvino"

    @classmethod
    def _register_builtins(cls) -> None:
        try:
            from .pytorch_engine import PyTorchEngine

            _try_register(cls.PYTORCH_ENGINE, PyTorchEngine)
        except ImportError:
            pass

        try:
            from .litert_engine import LiteRTEngine

            _try_register(cls.TFLITE_ENGINE, LiteRTEngine)
        except ImportError:
            pass

        try:
            from .tensorflow_engine import TensorFlowEngine

            _try_register(cls.TENSORFLOW_ENGINE, TensorFlowEngine)
        except ImportError:
            pass

        try:
            from .onnx_engine import ONNXEngine

            _try_register(cls.ONNX_ENGINE, ONNXEngine)
        except ImportError:
            pass

        try:
            from .openvino_engine import OpenVinoEngine

            _try_register(cls.OPENVINO_ENGINE, OpenVinoEngine)
        except ImportError:
            pass

    @staticmethod
    def register_engine(engine_type: str, engine_class: Type) -> None:
        _engine_registry[engine_type] = engine_class

    @staticmethod
    def create_engine(engine_type: str, device: str = "cpu"):
        # Correct call: use classmethod properly
        EngineFactory._register_builtins()  # ← This now works

        try:
            cls = _engine_registry[engine_type]
            return cls(device)
        except KeyError:
            raise ValueError(f"Unsupported engine type: {engine_type}")
