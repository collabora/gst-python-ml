# YoloTask
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

"""Backend-agnostic YOLO task.

`YoloTask` overrides `do_decode` with the YOLO-specific result handling
(per-instance class names, tracking ids, segmentation) expressed only through
the analytics facade and the engine result. It inherits `do_forward` from
`ObjectDetectorTask`. A backend element shell combines this mixin with its
framework element base (the gst backend uses `BaseObjectDetector`); the shell
supplies the engine wiring, the read-only `engine_name` property, and element
registration.

See `ObjectDetectorTask` for the host-element contract (self.engine, self.logger,
self.track). `do_decode`'s `buf` is the opaque metadata target.
"""

from backend import analytics
from tasks.object_detector import ObjectDetectorTask

COCO_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "TV",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}


class YoloTask(ObjectDetectorTask):
    """YOLO detection + tracking + segmentation, independent of any framework."""

    def do_decode(self, buf, result, stream_idx=0):
        self.logger.debug(
            f"Decoding YOLO result for buffer {hex(id(buf))}, stream {stream_idx}: {result}"
        )
        boxes = result.boxes
        masks = None
        if not self.engine.track:
            masks = result.masks

        if boxes is None or len(boxes) == 0:
            self.logger.info("No detections found.")
            return

        meta = analytics.add_relation_meta(buf)
        if not meta:
            self.logger.error(
                f"Stream {stream_idx} - Failed to add analytics relation metadata"
            )
            return

        self.logger.debug(
            f"Stream {stream_idx} - Attaching metadata for {len(boxes)} detections"
        )
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i]
            score = boxes.conf[i]
            label = boxes.cls[i]
            label_num = label.item()
            class_name = COCO_CLASSES.get(label_num, f"unknown_{label_num}")

            # Use class name for detection, track_id for tracking
            if self.engine.track and hasattr(boxes, "id") and boxes.id is not None:
                track_id = boxes.id[i]
                track_id_int = int(track_id.item())
                qk_string = f"stream_{stream_idx}_id_{track_id_int}"
            else:
                qk_string = (
                    f"stream_{stream_idx}_{class_name}"  # No index, just class name
                )

            od_mtd = analytics.add_object(
                meta,
                qk_string,
                x1.item(),
                y1.item(),
                x2.item() - x1.item(),
                y2.item() - y1.item(),
                score.item(),
            )
            if od_mtd is None:
                self.logger.error(
                    f"Stream {stream_idx} - Failed to add object detection metadata"
                )
                continue
            self.logger.debug(
                f"Stream {stream_idx} - Added od_mtd: label={qk_string}, x1={x1.item()}, y1={y1.item()}, w={x2.item()-x1.item()}, h={y2.item()-y1.item()}, score={score.item()}"
            )

            # Tracking metadata only when track=True
            if self.engine.track and hasattr(boxes, "id") and boxes.id is not None:
                tracking_mtd = analytics.add_tracking(meta, track_id_int)
                if tracking_mtd is None:
                    self.logger.error(
                        f"Stream {stream_idx} - Failed to add tracking metadata"
                    )
                    continue
                ret = analytics.relate(meta, od_mtd, tracking_mtd)
                if not ret:
                    self.logger.error(
                        f"Stream {stream_idx} - Failed to relate object detection and tracking metadata"
                    )
                else:
                    self.logger.debug(
                        f"Stream {stream_idx} - Linked od_mtd {od_mtd} to tracking_mtd {tracking_mtd}"
                    )

            if masks is not None:
                self.add_segmentation_metadata(buf, masks[i], x1, y1, x2, y2)

        attached_meta = analytics.get_relation_meta(buf)
        if attached_meta:
            count = analytics.relation_length(attached_meta)
            self.logger.info(
                f"Stream {stream_idx} - Metadata attached to buffer {hex(id(buf))}: {count} relations"
            )
        else:
            self.logger.error(
                f"Stream {stream_idx} - Metadata not attached to buffer after adding"
            )

    def add_segmentation_metadata(self, buf, mask, x1, y1, x2, y2):
        """
        Adds segmentation mask metadata to the buffer.
        """
        self.logger.info("Adding segmentation mask metadata")
        pass
