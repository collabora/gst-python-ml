# Yolo
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
from backend import GObject
import backend

CAN_REGISTER_ELEMENT = True
try:
    from base_objectdetector import BaseObjectDetector
    from tasks.yolo import YoloTask

    from engine.yolo_engine import YoloEngine
    from engine.engine_factory import EngineFactory

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(f"The 'yolo' element will not be available. Error {e}")


class YOLOTransform(BaseObjectDetector, YoloTask):
    """
    GStreamer element shell for YOLO model inference on video frames
    (detection, segmentation, and tracking). The result handling (do_decode)
    is inherited from the backend-agnostic YoloTask; this class supplies the
    engine wiring, the read-only engine_name property, and registration.
    """

    __gstmetadata__ = (
        "YOLO",
        "Transform",
        "Performs object detection, segmentation, and tracking using YOLO on video frames",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    def __init__(self):
        super().__init__()
        self.mgr.engine_name = "pyml_yolo_engine"
        EngineFactory.register(self.mgr.engine_name, YoloEngine)

    # make engine_name read only
    @GObject.Property(type=str)
    def engine_name(self):
        """Machine Learning Engine (read-only in this class)."""
        return self.mgr.engine_name

    @engine_name.setter
    def engine_name(self, value):
        raise ValueError(
            "The 'engine_name' property cannot be set in this derived class."
        )


# The class is backend-agnostic: under g2g the host imports this module and
# instantiates YOLOTransform directly, so no GObject registration applies.
# GStreamer factory registration runs only under the gst backend.
if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element("pyml_yolo", YOLOTransform)
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'pyml_yolo' element will not be registered because required modules are missing."
    )
