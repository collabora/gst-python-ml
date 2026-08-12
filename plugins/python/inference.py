# pyml_inference — generic passthrough for testing ML engines
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

from log.global_logger import GlobalLogger
import backend

CAN_REGISTER_ELEMENT = True
try:
    from video_transform import VideoTransform

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'pyml_inference' element will not be available. Error {e}"
    )


class GenericInferenceTransform(VideoTransform):
    """
    Generic passthrough element for testing any ML engine via the engine-name property.
    Runs do_forward() on each frame and logs the result. Buffer passes through unchanged.

    engine-name: pytorch (default), onnx, tensorflow, tflite, openvino

    Example:
      python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
        d. ! queue ! videoconvert ! videoscale \
        ! "video/x-raw,format=RGB,width=640,height=480" \
        ! pyml_inference engine-name=onnx model-name=yolo11m.onnx device=cpu \
        ! fakesink
    """

    __gstmetadata__ = (
        "Generic ML Inference",
        "Transform",
        "Passthrough element for testing ML engines; logs do_forward() output",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    def do_start(self):
        result = super().do_start()
        self.logger.info(
            f"pyml_inference started — engine={self.mgr.engine_name} "
            f"model={self.model_name} device={self.mgr.device}"
        )
        return result

    def process_frames(self, frames, num_sources, fmt, target):
        """Run the engine on the frame and log the result. The frame is unchanged."""
        frame = frames[0] if num_sources > 1 else frames

        if not self.engine:
            return

        result = self.engine.do_forward(frame)
        if result is not None:
            self.logger.info(f"inference result: {result}")


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element(
        "pyml_inference", GenericInferenceTransform
    )
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'pyml_inference' element will not be registered because required modules are missing."
    )
