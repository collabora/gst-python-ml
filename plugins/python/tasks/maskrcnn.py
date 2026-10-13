# MaskRCNNTask
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

from backend import analytics
from tasks.object_detector import ObjectDetectorTask


class MaskRCNNTask(ObjectDetectorTask):
    """Mask R-CNN result handling, independent of any framework."""

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
        meta = analytics.add_relation_meta(buf)
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

            # Use stream_idx in the quark string to differentiate streams
            qk_string = f"stream_{stream_idx}_{self.class_label(label)}"
            mtd = analytics.add_object(meta, qk_string, x1, y1, x2 - x1, y2 - y1, score)
            if mtd is not None:
                self.logger.info(
                    f"Stream {stream_idx} - Successfully added object detection metadata with quark {qk_string} and mtd {mtd}"
                )
            else:
                self.logger.error(
                    f"Stream {stream_idx} - Failed to add object detection metadata"
                )

        attached_meta = analytics.get_relation_meta(buf)
        if attached_meta:
            count = analytics.relation_length(attached_meta)
            self.logger.info(
                f"Stream {stream_idx} - Metadata successfully attached to buffer at address: {hex(id(buf))} with {count} relations"
            )
        else:
            self.logger.warning(
                f"Stream {stream_idx} - Failed to retrieve attached metadata immediately after addition for buffer: {hex(id(buf))}"
            )
