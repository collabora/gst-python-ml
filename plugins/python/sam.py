# SAM
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from log.global_logger import GlobalLogger
import backend

CAN_REGISTER_ELEMENT = True
try:
    from video_transform import VideoTransform
    from utils.format_converter import FormatConverter
    from engine.sam_engine import ExportedSam, SamEngine
    from engine.engine_factory import EngineFactory
    from backend import GObject
    from tasks.sam import SamTask

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(f"The 'sam' element will not be available. Error {e}")

# Header prefix for segmentation mask buffer metadata
SAM_META_HEADER = b"GST-SAM:"


class SamTransform(VideoTransform, SamTask):
    """
    GStreamer element for image segmentation using Segment Anything Model 2.

    Set model-name to a HuggingFace model ID, e.g.:
      facebook/sam2-hiera-large

    When visualize=True (default), colored mask overlays are drawn on the
    video frame. Mask metadata is always appended to the buffer as a
    GST-SAM: memory chunk (JSON with mask scores and shapes).
    """

    META_HEADER = SAM_META_HEADER

    __gstmetadata__ = (
        "SAM Segmentation",
        "Transform",
        "Image segmentation using Segment Anything Model 2",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    visualize = GObject.Property(
        type=bool,
        default=True,
        nick="Visualize Masks",
        blurb="Overlay colored segmentation masks on the video frame",
        flags=GObject.ParamFlags.READWRITE,
    )

    max_masks = GObject.Property(
        type=int,
        default=10,
        minimum=1,
        maximum=100,
        nick="Max Masks",
        blurb="Maximum number of segmentation masks to generate",
        flags=GObject.ParamFlags.READWRITE,
    )

    mode = GObject.Property(
        type=str,
        default="auto",
        nick="Segmentation Mode",
        blurb="Segmentation mode: 'auto' for automatic, 'points' for point-prompted",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        self.task_engine_name = self.mgr.engine_name = "mp_sam_engine"
        EngineFactory.register(self.mgr.engine_name, SamEngine)
        self.format_converter = FormatConverter()

    def export_model(self, model_name):
        return ExportedSam(model_name, self.mgr.engine_name)


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element("mp_sam", SamTransform)
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'mp_sam' element will not be registered because required modules are missing."
    )
