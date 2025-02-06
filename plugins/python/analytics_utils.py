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
            return metadata

        try:
            count = GstAnalytics.relation_get_length(meta)
            for index in range(count):
                ret, od_mtd = meta.get_od_mtd(index)
                if not ret or od_mtd is None:
                    continue

                label_quark = od_mtd.get_obj_type()
                label = GLib.quark_to_string(label_quark)
                track_id = self.extract_id_from_label(label)
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
        except Exception as e:
            self.logger.error(f"Error while extracting analytics metadata: {e}")
        return metadata

    def extract_id_from_label(self, label):
        """Extracts the numeric ID from a label formatted as 'id_<number>'."""
        match = re.match(r"id_(\d+)", label)
        if match:
            track_id = int(match.group(1))
            return track_id
        else:
            print("No ID found in label")  # Optional debug message for unmatched format
            return None  # Return None if the ID format is not found
