# MaskRCNN
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

from global_logger import GlobalLogger

CAN_REGISTER_ELEMENT = True
try:
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    gi.require_version("GstVideo", "1.0")
    gi.require_version("GstAnalytics", "1.0")
    gi.require_version("GLib", "2.0")
    from gi.repository import Gst, GObject, GstAnalytics, GLib  # noqa: E402
    import numpy as np

    from objectdetector_base import ObjectDetectorBase
except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'pyml_maskrcnn' element will not be available. Error {e}"
    )


class MaskRCNN(ObjectDetectorBase):
    """
    GStreamer element for Mask R-CNN model inference on video frames.
    """

    __gstmetadata__ = (
        "MaskRCNN",
        "Transform",
        "Applies the MaskRCNN object detection and segmentation model",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    def do_decode(self, buf, output, stream_idx=0):
        """
        Processes the Mask R-CNN model's output detections and adds metadata to the GStreamer buffer,
        tagged with the stream index.
        """
        boxes = output["boxes"]
        labels = output["labels"]
        scores = output["scores"]
        masks = output["masks"]  # Additional mask outputs for Mask R-CNN

        self.logger.info(
            f"Processing buffer at address: {hex(id(buf))} for stream {stream_idx}"
        )
        self.logger.info(f"Stream {stream_idx} - Processing {len(boxes)} detections")

        # Add analytics metadata to the buffer
        meta = GstAnalytics.buffer_add_analytics_relation_meta(buf)
        if not meta:
            self.logger.error(f"Stream {stream_idx} - Failed to add analytics metadata")
            return

        for i, (box, label, score, mask) in enumerate(
            zip(boxes, labels, scores, masks)
        ):
            x1, y1, x2, y2 = box
            self.logger.info(
                f"Stream {stream_idx} - Detection {i}: Box coordinates (x1={x1}, y1={y1}, x2={x2}, y2={y2}), "
                f"Label={label}, Score={score:.2f}"
            )

            # Convert mask to binary for further processing or metadata attachment
            binary_mask = (mask[0] > 0.5).astype(np.uint8)  # Threshold mask

            # Use stream_idx in the quark string to differentiate streams
            qk_string = f"stream_{stream_idx}_label_{label}"
            qk = GLib.quark_from_string(qk_string)
            ret, mtd = meta.add_od_mtd(qk, x1, y1, x2 - x1, y2 - y1, score)
            if ret:
                self.logger.info(
                    f"Stream {stream_idx} - Successfully added object detection metadata with quark {qk_string} and mtd {mtd}"
                )
            else:
                self.logger.error(
                    f"Stream {stream_idx} - Failed to add object detection metadata"
                )

        attached_meta = GstAnalytics.buffer_get_analytics_relation_meta(buf)
        if attached_meta:
            count = GstAnalytics.relation_get_length(attached_meta)
            self.logger.info(
                f"Stream {stream_idx} - Metadata successfully attached to buffer at address: {hex(id(buf))} with {count} relations"
            )
        else:
            self.logger.warning(
                f"Stream {stream_idx} - Failed to retrieve attached metadata immediately after addition for buffer: {hex(id(buf))}"
            )


if CAN_REGISTER_ELEMENT:
    GObject.type_register(MaskRCNN)
    __gstelementfactory__ = ("pyml_maskrcnn", Gst.Rank.NONE, MaskRCNN)
else:
    GlobalLogger().warning(
        "The 'pyml_maskrcnn' element will not be registered because required modules are missing."
    )
