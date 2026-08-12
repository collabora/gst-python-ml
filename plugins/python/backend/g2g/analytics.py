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
  * `add_*` return the sink's own staging handle, which `relate` passes back to
    pair a detection with its tracking id;
  * the sink stages straight into the host frame with no read path, so
    `read_objects` cannot see its own records back.
"""

import threading

from backend.analytics import AnalyticsBackend


class _RelationMeta:
    """The g2g stand-in for a buffer's relation meta: the bound sink plus a count
    of staged records (GstAnalytics tracks relations; the flat sink only counts)."""

    def __init__(self, sink):
        self.sink = sink
        self.count = 0


class _Bound:
    """What one element has staged: the frame's sink, and its label id space."""

    def __init__(self):
        self.sink = None
        self.meta = None
        self.labels = {}  # str -> u32 id
        self.next_id = 0
        self.published_names = -1


class G2gAnalyticsBackend(AnalyticsBackend):
    """`AnalyticsBackend` mapping detections/classifications onto a `MetaSink`."""

    def __init__(self):
        # Per-thread because the host runs one thread per element: one shared
        # binding would stage an element's detections on another's frame.
        self._threads = threading.local()

    @property
    def _bound(self):
        bound = getattr(self._threads, "bound", None)
        if bound is None:
            bound = _Bound()
            self._threads.bound = bound
        return bound

    def bind(self, sink):
        """Bind this frame's sink (called per frame by the g2g element bases).
        A fresh relation-meta is created lazily on first `add_relation_meta`."""
        bound = self._bound
        bound.sink = sink
        bound.meta = None
        # Each frame gets its own sink, so the names have to be sent again.
        bound.published_names = -1

    def quark(self, label):
        """Intern a string label into the `u32` id space the sink expects; ints
        pass through unchanged (matches the GStreamer GQuark contract)."""
        if isinstance(label, int):
            return label
        bound = self._bound
        qid = bound.labels.get(label)
        if qid is None:
            qid = bound.next_id
            bound.labels[label] = qid
            bound.next_id += 1
        return qid

    def _publish_class_names(self):
        """Send the interned label names to the sink, so a consumer can show a
        name instead of an id. Re-sent when a new label is interned mid-frame."""
        bound = self._bound
        if bound.sink is None or bound.next_id == bound.published_names:
            return
        names = [""] * bound.next_id
        for name, qid in bound.labels.items():
            names[qid] = name
        bound.sink.set_class_names(names)
        bound.published_names = bound.next_id

    def add_relation_meta(self, buf):
        bound = self._bound
        if bound.sink is None:
            return None
        if bound.meta is None:
            bound.meta = _RelationMeta(bound.sink)
        return bound.meta

    def get_relation_meta(self, buf):
        return self._bound.meta

    def remove_relation_meta(self, buf):
        bound = self._bound
        had = bound.meta is not None
        bound.meta = None
        return had

    def relation_length(self, meta):
        return meta.count if meta else 0

    def add_object(self, meta, label, x, y, w, h, score):
        if meta is None:
            return None
        meta.count += 1
        qid = self.quark(label)
        self._publish_class_names()
        return meta.sink.add_object(
            qid, float(x), float(y), float(w), float(h), float(score)
        )

    def add_classification(self, meta, index, label):
        if meta is None:
            return None
        # The flat sink's add_classification is (label, score); the gst `index`
        # (stream id) has no place in it and is dropped.
        meta.count += 1
        qid = self.quark(label)
        self._publish_class_names()
        return meta.sink.add_classification(qid, 1.0)

    def add_tracking(self, meta, track_id, timestamp=None):
        # The host stamps its own arrival time, so the gst `timestamp` is dropped.
        if meta is None:
            return None
        meta.count += 1
        return meta.sink.add_tracking(int(track_id))

    def relate(self, meta, src, dst):
        if meta is None or src is None or dst is None:
            return False
        meta.sink.relate(int(src), int(dst))
        return True

    def read_objects(self, meta):
        # The sink is write-only (staged straight into the host frame); the
        # element cannot read its own staged detections back.
        return []


#: Defined here, like `frameio`, to avoid a circular import through `backend`.
analytics = G2gAnalyticsBackend()
