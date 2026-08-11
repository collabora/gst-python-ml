# G2gAnalyticsBackend (g2g backend)
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Library General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.

"""Analytics metadata over the g2g host's `MetaSink`.

GStreamer attaches a `GstAnalyticsRelationMeta` to the buffer and relates
detections, classifications and tracking records. The g2g host instead hands the
element a flat, write-only `MetaSink` per frame with `add_object(label, x, y, w,
h, score)` / `add_classification(label, score)` / `add_blob(...)`; the host then
materializes those into the frame's `AnalyticsMeta`. This backend maps the rich
`AnalyticsBackend` interface the leaf task code uses onto that flat sink:

  * the "relation meta" handle is a thin wrapper over the bound sink that counts
    how many records were staged (so `relation_length` works);
  * string labels are interned to the `u32` ids the sink expects (`quark`);
  * tracking relations and read-back are not represented by the flat sink, so
    `add_tracking` / `relate` / `read_objects` are no-ops for now.
"""

from backend.analytics import AnalyticsBackend


class _RelationMeta:
    """The g2g stand-in for a buffer's relation meta: the bound sink plus a count
    of staged records (GstAnalytics tracks relations; the flat sink only counts)."""

    def __init__(self, sink):
        self.sink = sink
        self.count = 0


class G2gAnalyticsBackend(AnalyticsBackend):
    """`AnalyticsBackend` mapping detections/classifications onto a `MetaSink`."""

    def __init__(self):
        self._sink = None
        self._meta = None
        self._labels = {}  # str -> u32 id
        self._next_id = 0

    def bind(self, sink):
        """Bind this frame's sink (called per frame by the g2g element bases).
        A fresh relation-meta is created lazily on first `add_relation_meta`."""
        self._sink = sink
        self._meta = None

    def quark(self, label):
        """Intern a string label into the `u32` id space the sink expects; ints
        pass through unchanged (matches the GStreamer GQuark contract)."""
        if isinstance(label, int):
            return label
        qid = self._labels.get(label)
        if qid is None:
            qid = self._next_id
            self._labels[label] = qid
            self._next_id += 1
        return qid

    def add_relation_meta(self, buf):
        if self._sink is None:
            return None
        if self._meta is None:
            self._meta = _RelationMeta(self._sink)
        return self._meta

    def get_relation_meta(self, buf):
        return self._meta

    def remove_relation_meta(self, buf):
        had = self._meta is not None
        self._meta = None
        return had

    def relation_length(self, meta):
        return meta.count if meta else 0

    def add_object(self, meta, label, x, y, w, h, score):
        if meta is None:
            return None
        meta.sink.add_object(
            self.quark(label), float(x), float(y), float(w), float(h), float(score)
        )
        meta.count += 1
        # Opaque handle: the record's index (used only as a relate() endpoint,
        # which the flat sink does not support).
        return meta.count - 1

    def add_classification(self, meta, index, label):
        if meta is None:
            return None
        # The flat sink's add_classification is (label, score); the gst `index`
        # (stream id) has no place in it and is dropped.
        meta.sink.add_classification(self.quark(label), 1.0)
        meta.count += 1
        return meta.count - 1

    def add_tracking(self, meta, track_id, timestamp=None):
        # The flat g2g sink stages no tracking records yet.
        return None

    def relate(self, meta, src, dst):
        # No relation graph in the flat sink.
        return False

    def read_objects(self, meta):
        # The sink is write-only (staged straight into the host frame); the
        # element cannot read its own staged detections back.
        return []


#: The analytics implementation for this backend (shared singleton; the element
#: bases bind the per-frame sink on it). Defined here, like `frameio`, to avoid a
#: circular import through `backend`.
analytics = G2gAnalyticsBackend()
