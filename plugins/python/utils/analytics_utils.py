# AnalyticsUtils
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

ANALYTICS_UTILS_AVAILABLE = True
try:
    import re
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GLib", "2.0")
    gi.require_version("GstAnalytics", "1.0")
    from gi.repository import GstAnalytics, GLib  # noqa: E402

    from log.logger_factory import LoggerFactory  # noqa: E402
except (ImportError, ValueError):
    ANALYTICS_UTILS_AVAILABLE = False


class AnalyticsUtils:
    """
    Utility class for extracting analytics metadata from GStreamer buffers.

    This class provides methods to retrieve and parse object detection (OD) and tracking metadata
    attached to video buffers using GStreamer's GstAnalytics API. It is designed to work with
    metadata added by object detection and tracking elements, such as YOLO-based transformers.

    Supported Label Formats:
    - Advanced person tracking: 'stream_<idx>_person_id_<id>' → label='person', track_id=<id>
    - Advanced ball tracking: 'stream_<idx>_ball_id_<id>' → label='ball', track_id=<id>
    - Simpler tracking: 'stream_<idx>_id_<id>' → label='id_<id>', track_id=<id>
    - Class name (no tracking): 'stream_<idx>_<class_name>' → label='<class_name>', track_id=None
    - Fallback: Any other format → label='<full_label>', track_id=None

    The extracted metadata includes labels, track IDs, confidence scores, and bounding box coordinates.
    """

    def __init__(self):
        super().__init__()
        self.logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)

    def extract_analytics_metadata(self, buffer):
        metadata = []
        meta = GstAnalytics.buffer_get_analytics_relation_meta(buffer)
        if not meta:
            self.logger.info("No analytics relation metadata found on buffer")
            return metadata

        count = GstAnalytics.relation_get_length(meta)
        self.logger.info(f"Found {count} analytics relations in metadata")
        for index in range(count):
            ret, od_mtd = meta.get_od_mtd(index)
            if not ret or od_mtd is None:
                continue
            label_quark = od_mtd.get_obj_type()
            full_label = GLib.quark_to_string(label_quark)
            self.logger.debug(f"Index {index}: quark={full_label}")

            # CLIP/SigLIP zero-shot classification metadata (clip_<label>)
            clip_match = re.match(r"clip_(.+)", full_label)
            if clip_match:
                label = clip_match.group(1).replace("_", " ")
                location = od_mtd.get_location()
                _, _x, _y, _w, _h, confidence = location
                metadata.append(
                    {
                        "type": "classification",
                        "label": label,
                        "confidence": confidence,
                    }
                )
                self.logger.debug(f"Added classification entry: {metadata[-1]}")
                continue

            # Standard object detection metadata
            track_id, label = self.extract_id_from_label(full_label)
            location = od_mtd.get_location()
            presence, x, y, w, h, loc_conf_lvl = location
            if presence:
                metadata.append(
                    {
                        "label": label,
                        "track_id": track_id,
                        "confidence": loc_conf_lvl,
                        "box": {"x1": x, "y1": y, "x2": x + w, "y2": y + h},
                    }
                )
                self.logger.debug(f"Added detection entry: {metadata[-1]}")
        return metadata

    def extract_id_from_label(self, full_label):
        # Handle advanced person format: stream_\d+_person_id_\d+
        match = re.match(r"stream_\d+_person_id_(\d+)", full_label)
        if match:
            track_id = int(match.group(1))
            label = "person"
            self.logger.debug(
                f"Extracted track_id {track_id} and label '{label}' from '{full_label}' (advanced person)"
            )
            return track_id, label

        # Handle advanced ball format: stream_\d+_ball_id_\d+
        match = re.match(r"stream_\d+_ball_id_(\d+)", full_label)
        if match:
            track_id = int(match.group(1))
            label = "ball"
            self.logger.debug(
                f"Extracted track_id {track_id} and label '{label}' from '{full_label}' (advanced ball)"
            )
            return track_id, label

        # Fallback to simpler tracking format: stream_\d+_id_\d+
        match = re.match(r"stream_\d+_id_(\d+)", full_label)
        if match:
            track_id = int(match.group(1))
            label = f"id_{track_id}"
            self.logger.debug(
                f"Extracted track_id {track_id} and label '{label}' from '{full_label}' (simpler tracking)"
            )
            return track_id, label

        # Fallback to class name format: stream_\d+_(.+)
        match = re.match(r"stream_\d+_(.+)", full_label)
        if match:
            class_name = match.group(1)
            label = class_name  # Use class name directly
            self.logger.debug(
                f"Extracted class label '{label}' from '{full_label}' (class name)"
            )
            return None, label

        self.logger.info(f"No recognizable format in label '{full_label}', using as-is")
        return None, full_label
