# SAM
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
    from utils.format_converter import FormatConverter
    from engine.sam_engine import SamEngine
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
        self.mgr.engine_name = "pyml_sam_engine"
        EngineFactory.register(self.mgr.engine_name, SamEngine)
        self.format_converter = FormatConverter()

    @GObject.Property(type=str)
    def engine_name(self):
        """Machine Learning Engine (read-only for this element)."""
        return self.mgr.engine_name

    @engine_name.setter
    def engine_name(self, value):
        raise ValueError("'engine_name' is read-only for pyml_sam")


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element("pyml_sam", SamTransform)
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'pyml_sam' element will not be registered because required modules are missing."
    )
