# ObjectDetector
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

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
        "mp_objectdetector", ObjectDetector
    )
