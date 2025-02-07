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

    # @GObject.Property
    # def model_name(self):
    #     return "maskrcnn_resnet50_fpn"  # Always return the correct model name

    # @model_name.setter
    # def model_name(self, value):
    #     # Emit a warning if someone tries to set the model-name property
    #     self.logger.warning(
    #         f"Attempt to change the model-name property to '{value}' is not allowed. "
    #         "MaskRCNN will always use 'maskrcnn_resnet50_fpn'."
    #     )

    def do_decode(self, buf, output):
        """
        Processes the Mask R-CNN model's output detections
        and adds metadata to the GStreamer buffer.
        """
        boxes = output["boxes"]
        labels = output["labels"]
        scores = output["scores"]
        masks = output["masks"]  # Additional mask outputs for Mask R-CNN

        self.logger.info(f"Processing buffer at address: {hex(id(buf))}")
        self.logger.info(f"Processing {len(boxes)} detections.")

        for i, (box, label, score, mask) in enumerate(
            zip(boxes, labels, scores, masks)
        ):
            x1, y1, x2, y2 = box
            self.logger.info(
                f"Detection {i}: Box coordinates (x1={x1}, y1={y1}, x2={x2}, y2={y2}), "
                f"Label={label}, Score={score:.2f}"
            )

            # Convert mask to binary for further processing or metadata attachment
            binary_mask = (mask[0] > 0.5).astype(np.uint8)  # Threshold mask

            meta = GstAnalytics.buffer_add_analytics_relation_meta(buf)
            if meta:
                qk = GLib.quark_from_string(f"label_{label}")
                ret, mtd = meta.add_od_mtd(qk, x1, y1, x2 - x1, y2 - y1, score)
                if ret:
                    self.logger.info(
                        f"Successfully added object detection metadata with quark {qk} and mtd {mtd}"
                    )
                else:
                    self.logger.error("Failed to add object detection metadata")
            else:
                self.logger.error("Failed to add GstAnalytics metadata to buffer")

        attached_meta = GstAnalytics.buffer_get_analytics_relation_meta(buf)
        if attached_meta:
            self.logger.info(
                f"Metadata successfully attached to buffer at address: {hex(id(buf))}"
            )
        else:
            self.logger.warning(
                f"Failed to retrieve attached metadata immediately after addition for buffer: {hex(id(buf))}"
            )


if CAN_REGISTER_ELEMENT:
    GObject.type_register(MaskRCNN)
    __gstelementfactory__ = ("pyml_maskrcnn", Gst.Rank.NONE, MaskRCNN)
else:
    GlobalLogger().warning(
        "The 'pyml_maskrcnn' element will not be registered because required modules are missing."
    )
