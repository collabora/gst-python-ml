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

try:
    from .pytorch_engine import PyTorchEngine

    _pytorch_engine_available = True
except ImportError:
    _pytorch_engine_available = False

try:
    from .pytorch_yolo_engine import PyTorchYoloEngine

    _pytorch_yolo_engine_available = True
except ImportError:
    _pytorch_yolo_engine_available = False

try:
    from .litert_engine import LiteRTEngine

    _tflite_engine_available = True
except ImportError:
    _tflite_engine_available = False

try:
    from .tensorflow_engine import TensorFlowEngine

    _tensorflow_engine_available = True
except ImportError:
    _tensorflow_engine_available = False

try:
    from .onnx_engine import ONNXEngine

    _onnx_engine_available = True
except ImportError:
    _onnx_engine_available = False

try:
    from .openvino_engine import OpenVinoEngine

    _openvino_engine_available = True
except ImportError:
    _openvino_engine_available = False


class EngineFactory:
    # Define the constant strings for each engine
    PYTORCH_ENGINE = "pytorch"
    PYTORCH_YOLO_ENGINE = "pytorch-yolo"
    TFLITE_ENGINE = "tflite"
    TENSORFLOW_ENGINE = "tensorflow"
    ONNX_ENGINE = "onnx"
    OPENVINO_ENGINE = "openvino"

    @staticmethod
    def create_engine(engine_type, device="cpu"):
        """
        Factory method to create the appropriate engine based on the engine type.
        :param engine_type: The type of the ML engine, e.g., "pytorch" or "tflite".
        :param device: The device to run the engine on (default is "cpu").
        :return: An instance of the appropriate ML engine class.
        """
        if engine_type == EngineFactory.PYTORCH_ENGINE:
            if _pytorch_engine_available:
                return PyTorchEngine(device)
            else:
                raise ImportError(
                    f"{EngineFactory.PYTORCH_ENGINE} engine is not available."
                )

        if engine_type == EngineFactory.PYTORCH_YOLO_ENGINE:
            if _pytorch_yolo_engine_available:
                return PyTorchYoloEngine(device)
            else:
                raise ImportError(
                    f"{EngineFactory.PYTORCH_YOLO_ENGINE} engine is not available."
                )

        elif engine_type == EngineFactory.TFLITE_ENGINE:
            if _tflite_engine_available:
                return LiteRTEngine(device)
            else:
                raise ImportError(
                    f"{EngineFactory.TFLITE_ENGINE} engine is not available."
                )

        elif engine_type == EngineFactory.TENSORFLOW_ENGINE:
            if _tensorflow_engine_available:
                return TensorFlowEngine(device)
            else:
                raise ImportError(
                    f"{EngineFactory.TENSORFLOW_ENGINE} engine is not available."
                )

        elif engine_type == EngineFactory.ONNX_ENGINE:
            if _onnx_engine_available:
                return ONNXEngine(device)
            else:
                raise ImportError(
                    f"{EngineFactory.ONNX_ENGINE} engine is not available."
                )

        elif engine_type == EngineFactory.OPENVINO_ENGINE:
            if _openvino_engine_available:
                return OpenVinoEngine(device)
            else:
                raise ImportError(
                    f"{EngineFactory.OPENVINO_ENGINE} engine is not available."
                )

        else:
            raise ValueError(f"Unsupported engine type: {engine_type}")
