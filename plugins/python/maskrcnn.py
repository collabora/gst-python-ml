# MaskRCNN
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from log.global_logger import GlobalLogger
import backend

CAN_REGISTER_ELEMENT = True
try:
    from base_objectdetector import BaseObjectDetector
    from tasks.maskrcnn import MaskRCNNTask
except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'mp_maskrcnn' element will not be available. Error {e}"
    )


class MaskRCNN(BaseObjectDetector, MaskRCNNTask):
    """
    GStreamer element for Mask R-CNN model inference on video frames.
    """

    __gstmetadata__ = (
        "MaskRCNN",
        "Transform",
        "Applies the MaskRCNN object detection and segmentation model",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element("mp_maskrcnn", MaskRCNN)
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'mp_maskrcnn' element will not be registered because required modules are missing."
    )
