# RelationMetaAnalytics
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

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstAnalytics", "1.0")
gi.require_version("GLib", "2.0")
from gi.repository import Gst, GstAnalytics, GLib  # noqa: E402

from backend.analytics import AnalyticsBackend  # noqa: E402


class RelationMetaAnalytics(AnalyticsBackend):
    def add_relation_meta(self, buf):
        return GstAnalytics.buffer_add_analytics_relation_meta(buf)

    def get_relation_meta(self, buf):
        return GstAnalytics.buffer_get_analytics_relation_meta(buf)

    def remove_relation_meta(self, buf):
        meta = GstAnalytics.buffer_get_analytics_relation_meta(buf)
        if meta is None:
            return False
        # GstAnalytics relation meta is an add-once accumulator: detections are
        # added into it as *_mtd entries. The Python bindings expose no path to
        # detach it (the RelationMeta wrapper is not a Gst.Meta, so it cannot be
        # passed to Gst.Buffer.remove_meta). Attempt it for bindings that do
        # accept it; otherwise report that nothing was removed.
        try:
            return bool(buf.remove_meta(meta))
        except (TypeError, AttributeError):
            return False

    def relation_length(self, meta):
        return GstAnalytics.relation_get_length(meta)

    def quark(self, label):
        if isinstance(label, int):
            return label
        return GLib.quark_from_string(str(label))

    def add_object(self, meta, label, x, y, w, h, score):
        qk = self.quark(label)
        ret, mtd = meta.add_od_mtd(qk, x, y, w, h, score)
        return mtd if ret else None

    def add_classification(self, meta, index, label):
        qk = self.quark(label)
        ret, mtd = meta.add_one_cls_mtd(index, qk)
        return mtd if ret else None

    def add_tracking(self, meta, track_id, timestamp=None):
        if timestamp is None:
            timestamp = Gst.util_get_timestamp()
        ret, mtd = meta.add_tracking_mtd(track_id, timestamp)
        return mtd if ret else None

    def relate(self, meta, src, dst):
        return bool(
            GstAnalytics.RelationMeta.set_relation(
                meta, GstAnalytics.RelTypes.RELATE_TO, src.id, dst.id
            )
        )

    def read_objects(self, meta):
        objects = []
        count = GstAnalytics.relation_get_length(meta)
        for index in range(count):
            ret, od_mtd = meta.get_od_mtd(index)
            if not ret or od_mtd is None:
                continue
            label = GLib.quark_to_string(od_mtd.get_obj_type())
            presence, x, y, w, h, score = od_mtd.get_location()
            if not presence:
                continue
            objects.append(
                {"label": label, "x": x, "y": y, "w": w, "h": h, "score": score}
            )
        return objects
