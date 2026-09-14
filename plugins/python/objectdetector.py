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

from base_objectdetector import BaseObjectDetector
from backend import GObject
import backend


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


# The class is backend-agnostic: under g2g the host imports this module and
# instantiates ObjectDetector directly, so no GObject registration applies.
# GStreamer factory registration runs only under the gst backend.
if backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element(
        "pyml_objectdetector", ObjectDetector
    )
