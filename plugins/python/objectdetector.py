# ObjectDetector
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

CAN_REGISTER_ELEMENT = True
try:
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    gi.require_version("GLib", "2.0")
    from gi.repository import Gst, GObject  # noqa: E402
    from base_objectdetector import BaseObjectDetector
except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'objectdetector_pylm' element will not be available. Error: {e}"
    )


class ObjectDetector(BaseObjectDetector):
    """
    GStreamer element for a general object detector where the user sets the model-name property.
    """

    __gstmetadata__ = (
        "ObjectDetector",
        "Transform",
        "General purpose object",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    confidence = GObject.Property(
        type=float,
        default=0.25,
        minimum=0.0,
        maximum=1.0,
        nick="Confidence Threshold",
        blurb="Minimum detection confidence for the decoder post-process "
        "(anchor_free); lower = more (and weaker) detections",
        flags=GObject.ParamFlags.READWRITE,
    )
    nms_iou = GObject.Property(
        type=float,
        default=0.45,
        minimum=0.0,
        maximum=1.0,
        nick="NMS IoU",
        blurb="NMS IoU threshold for the decoder post-process; higher keeps "
        "more overlapping boxes",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        self.logger.info(
            "ObjectDetector created without a model. Please set the 'model-name' property."
        )

    def do_forward(self, frames):
        # Push decoder thresholds to the engine before it post-processes.
        if self.engine:
            self.engine.conf = self.confidence
            self.engine.iou = self.nms_iou
        return super().do_forward(frames)


if CAN_REGISTER_ELEMENT:
    GObject.type_register(ObjectDetector)
    __gstelementfactory__ = ("pyml_objectdetector", Gst.Rank.NONE, ObjectDetector)
else:
    GlobalLogger().warning(
        "The 'pyml_objectdetector' element will not be registered because base_objectdetector module is missing."
    )
