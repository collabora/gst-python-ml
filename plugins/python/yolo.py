# Yolo
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
    from objectdetector_base import ObjectDetectorBase
except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(f"The 'yolo' element will not be available. Error {e}")

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


class YOLOTransform(ObjectDetectorBase):
    """
    GStreamer element for YOLO model inference on video frames
    (detection, segmentation, and tracking).
    """

    __gstmetadata__ = (
        "YOLO",
        "Transform",
        "Performs object detection, segmentation, and tracking using YOLO on video frames",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    def __init__(self):
        super().__init__()
        self.engine_name = "pytorch-yolo"

    def do_decode(self, buf, result):
        """
        Decode the YOLO model's output detections (and optionally segmentation masks)
        and add metadata to the GStreamer buffer.
        """
        # Extract relevant data from the result
        boxes = result.boxes  # Extract boxes object
        masks = None
        if not self.engine.track:
            masks = result.masks  # Extract masks for segmentation (if available)

        if boxes is None or len(boxes) == 0:
            self.logger.info("No detections found.")
            return

        # Iterate over the detected boxes and add metadata
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i]  # Extract bounding box coordinates
            score = boxes.conf[i]  # Extract confidence score
            label = boxes.cls[i]  # Extract class label

            # Add object detection metadata
            meta = GstAnalytics.buffer_add_analytics_relation_meta(buf)
            if meta:
                label_num = label.item()
                qk_string = COCO_CLASSES.get(
                    label_num, f"unknown_{label_num}"
                )  # Default to 'unknown' if label is not found
                # Handle tracking if enabled
                tracking_mtd = None
                if self.engine.track:
                    track_id = result.boxes.id[i]  # Extract track ID
                    if track_id is not None:
                        ret, tracking_mtd = meta.add_tracking_mtd(
                            track_id, Gst.util_get_timestamp()
                        )
                        if not ret:
                            self.logger.error("Failed to add tracking metadata")

                        track_id_int = int(track_id.item())
                        # self.logger.info(f"Track ID {track_id_int} found for object {i}")
                        qk_string = f"id_{track_id_int}"

                qk = GLib.quark_from_string(qk_string)
                ret, od_mtd = meta.add_od_mtd(
                    qk,
                    x1.item(),
                    y1.item(),
                    x2.item() - x1.item(),
                    y2.item() - y1.item(),
                    score.item(),
                )
                if not ret:
                    self.logger.error("Failed to add object detection metadata")

                if tracking_mtd is not None:
                    ret = GstAnalytics.RelationMeta.set_relation(
                        meta,
                        GstAnalytics.RelTypes.RELATE_TO,
                        od_mtd.id,
                        tracking_mtd.id,
                    )
                    if not ret:
                        self.logger.error(
                            "Failed to relate object detection and tracking meta data"
                        )

            # Handle segmentation masks (if available)
            if masks is not None:
                self.add_segmentation_metadata(buf, masks[i], x1, y1, x2, y2)

        # Log buffer state after metadata attachment
        attached_meta = GstAnalytics.buffer_get_analytics_relation_meta(buf)
        if not attached_meta:
            self.logger.warning(
                f"Failed to retrieve attached metadata immediately after addition for buffer: {hex(id(buf))}"
            )

    def add_segmentation_metadata(self, buf, mask, x1, y1, x2, y2):
        """
        Adds segmentation mask metadata to the buffer.
        """
        self.logger.info("Adding segmentation mask metadata")
        pass


if CAN_REGISTER_ELEMENT:
    GObject.type_register(YOLOTransform)
    __gstelementfactory__ = ("pyml_yolo", Gst.Rank.NONE, YOLOTransform)
else:
    GlobalLogger().warning(
        "The 'pyml_yolo' element will not be registered because required modules are missing."
    )
