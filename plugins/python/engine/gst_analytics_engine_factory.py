# GstAnalyticsEngineFactory
# Copyright (C) 2024 Collabora Ltd.
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
    from .gst_analytics_pytorch_engine import GstAnalyticsPyTorchEngine

    _pytorch_engine_available = True
except ImportError:
    _pytorch_engine_available = False

try:
    from .gst_analytics_pytorch_yolo_engine import GstAnalyticsPyTorchYoloEngine

    _pytorch_yolo_engine_available = True
except ImportError:
    _pytorch_yolo_engine_available = False

try:
    from .gst_analytics_tflite_engine import GstAnalyticsTFLiteEngine

    _tflite_engine_available = True
except ImportError:
    _tflite_engine_available = False

try:
    from .gst_analytics_tensorflow_engine import GstAnalyticsTensorFlowEngine

    _tensorflow_engine_available = True
except ImportError:
    _tensorflow_engine_available = False

try:
    from .gst_analytics_onnx_engine import GstAnalyticsONNXEngine

    _onnx_engine_available = True
except ImportError:
    _onnx_engine_available = False

try:
    from .gst_analytics_openvino_engine import GstAnalyticsOpenVinoEngine

    _openvino_engine_available = True
except ImportError:
    _openvino_engine_available = False


class GstAnalyticsEngineFactory:
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
        if engine_type == GstAnalyticsEngineFactory.PYTORCH_ENGINE:
            if _pytorch_engine_available:
                return GstAnalyticsPyTorchEngine(device)
            else:
                raise ImportError(
                    f"{GstAnalyticsEngineFactory.PYTORCH_ENGINE} engine is not available."
                )

        if engine_type == GstAnalyticsEngineFactory.PYTORCH_YOLO_ENGINE:
            if _pytorch_yolo_engine_available:
                return GstAnalyticsPyTorchYoloEngine(device)
            else:
                raise ImportError(
                    f"{GstAnalyticsEngineFactory.PYTORCH_YOLO_ENGINE} engine is not available."
                )

        elif engine_type == GstAnalyticsEngineFactory.TFLITE_ENGINE:
            if _tflite_engine_available:
                return GstAnalyticsTFLiteEngine(device)
            else:
                raise ImportError(
                    f"{GstAnalyticsEngineFactory.TFLITE_ENGINE} engine is not available."
                )

        elif engine_type == GstAnalyticsEngineFactory.TENSORFLOW_ENGINE:
            if _tensorflow_engine_available:
                return GstAnalyticsTensorFlowEngine(device)
            else:
                raise ImportError(
                    f"{GstAnalyticsEngineFactory.TENSORFLOW_ENGINE} engine is not available."
                )

        elif engine_type == GstAnalyticsEngineFactory.ONNX_ENGINE:
            if _onnx_engine_available:
                return GstAnalyticsONNXEngine(device)
            else:
                raise ImportError(
                    f"{GstAnalyticsEngineFactory.ONNX_ENGINE} engine is not available."
                )

        elif engine_type == GstAnalyticsEngineFactory.OPENVINO_ENGINE:
            if _openvino_engine_available:
                return GstAnalyticsOpenVinoEngine(device)
            else:
                raise ImportError(
                    f"{GstAnalyticsEngineFactory.OPENVINO_ENGINE} engine is not available."
                )

        else:
            raise ValueError(f"Unsupported engine type: {engine_type}")
