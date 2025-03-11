# AnalyticsUtils
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

ANALYTICS_UTILS_AVAILABLE = True
try:
    import re
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GLib", "2.0")
    gi.require_version("GstAnalytics", "1.0")
    from gi.repository import Gst, GstAnalytics, GLib  # noqa: E402

    from log.logger_factory import LoggerFactory
except ImportError:
    ANALYTICS_UTILS_AVAILABLE = False


class AnalyticsUtils:
    def __init__(self):
        super().__init__()
        self.logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)

    def extract_analytics_metadata(self, buffer):
        metadata = []
        meta = GstAnalytics.buffer_get_analytics_relation_meta(buffer)
        if not meta:
            self.logger.info("No analytics relation metadata found on buffer")
            return metadata

        try:
            count = GstAnalytics.relation_get_length(meta)
            self.logger.info(f"Found {count} analytics relations in metadata")
            for index in range(count):
                ret, od_mtd = meta.get_od_mtd(index)
                if not ret or od_mtd is None:
                    # self.logger.warning(f"Failed to get od_mtd at index {index}")
                    continue
                label_quark = od_mtd.get_obj_type()
                full_label = GLib.quark_to_string(label_quark)
                self.logger.debug(f"Index {index}: quark={full_label}")
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
                    self.logger.debug(f"Added metadata entry: {metadata[-1]}")
        except Exception as e:
            self.logger.error(f"Error while extracting analytics metadata: {e}")
        return metadata

    def extract_id_from_label(self, full_label):
        match = re.match(r"stream_\d+_id_(\d+)", full_label)
        if match:
            track_id = int(match.group(1))
            label = f"id_{track_id}"
            self.logger.debug(
                f"Extracted track_id {track_id} and label '{label}' from '{full_label}'"
            )
            return track_id, label
        match = re.match(
            r"stream_\d+_(.+)", full_label
        )  # Match class name after stream_<idx>_
        if match:
            class_name = match.group(1)
            label = class_name  # Use class name directly
            self.logger.debug(f"Extracted class label '{label}' from '{full_label}'")
            return None, label
        self.logger.info(f"No recognizable format in label '{full_label}', using as-is")
        return None, full_label
