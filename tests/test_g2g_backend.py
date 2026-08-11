"""Unit tests for the g2g element backend (`GSTML_BACKEND=g2g`).

These exercise the backend with no GStreamer present: the backend selection, the
`GObject` / `FlowReturn` shims, the `G2gFrameIO` buffer round-trip, the
`G2gAnalyticsBackend` mapping onto a flat sink, and an end-to-end `g2g_process`
on a `VideoTransform` subclass. The g2g host's `FrameBuffer` (a writable
buffer-protocol view) and `MetaSink` (write-only staging) are stubbed with a
`bytearray` and a recording object, so no Rust host is needed.
"""

import os
import sys
from pathlib import Path

import numpy as np

# Select the g2g backend before importing `backend`, and make the plugin package
# importable.
os.environ["GSTML_BACKEND"] = "g2g"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "plugins" / "python"))

import backend  # noqa: E402
from backend import (
    GObject,
    FlowReturn,
    frameio,
    analytics,
    VideoTransform,
)  # noqa: E402


class StubMetaSink:
    """Stand-in for the host's write-only `g2g.MetaSink`.

    Like the real sink, every `add_*` stages one record into a single list and
    returns its index, which is the handle `relate` takes.
    """

    def __init__(self):
        self.staged = []
        self.relations = []

    def _stage(self, record):
        self.staged.append(record)
        return len(self.staged) - 1

    def add_object(self, label, x, y, w, h, score):
        return self._stage(("object", label, x, y, w, h, score))

    def add_classification(self, label, score):
        return self._stage(("classification", label, score))

    def add_blob(self, header, payload):
        return self._stage(("blob", header, payload))

    def add_tracking(self, object_id):
        return self._stage(("tracking", object_id))

    def relate(self, src, dst):
        self.relations.append((src, dst))

    def _of_kind(self, kind):
        return [record[1:] for record in self.staged if record[0] == kind]

    @property
    def objects(self):
        return self._of_kind("object")

    @property
    def blobs(self):
        return self._of_kind("blob")


def test_backend_selected_is_g2g():
    assert backend.BACKEND == "g2g"
    assert FlowReturn.OK == 0
    assert FlowReturn.ERROR != FlowReturn.OK


def test_gobject_property_shim_decorator_and_attribute_forms():
    class Widget:
        @GObject.Property(type=str)
        def name(self):
            return getattr(self, "_n", "default")

        @name.setter
        def name(self, value):
            self._n = value.upper()

        size = GObject.Property(type=int, default=7, nick="Size", blurb="px")

    w = Widget()
    assert w.name == "default"  # getter default
    w.name = "yolo"
    assert w.name == "YOLO"  # setter ran
    assert w.size == 7  # attribute-form default
    w.size = 42
    assert w.size == 42


def test_frameio_read_write_round_trip():
    width, height = 4, 3
    buf = bytearray(width * height * 3)  # RGB, writable buffer-protocol object
    sink = StubMetaSink()
    frameio.bind(sink, "RGB")

    frame, num_sources, fmt = frameio.read_frames(buf, None, width, height)
    assert num_sources == 1 and fmt == "RGB"
    assert frame.shape == (height, width, 3)

    frameio.write_frame(buf, np.full((height, width, 3), 200, dtype=np.uint8))
    assert all(b == 200 for b in buf), "write_frame must update the buffer in place"

    frameio.append_blob(buf, "tag", b"\x01\x02")
    assert sink.blobs == [("tag", b"\x01\x02")]


def test_analytics_maps_onto_flat_sink():
    sink = StubMetaSink()
    analytics.bind(sink)

    meta = analytics.add_relation_meta(buf=None)
    assert meta is not None
    assert analytics.get_relation_meta(None) is meta

    # String labels intern to stable u32 ids (quark), reused across calls.
    analytics.add_object(meta, "person", 1, 2, 3, 4, 0.9)
    analytics.add_object(meta, "person", 5, 6, 7, 8, 0.8)
    analytics.add_object(meta, "handbag", 0, 0, 1, 1, 0.5)

    assert analytics.relation_length(meta) == 3
    labels = [o[0] for o in sink.objects]
    assert labels[0] == labels[1], "same string -> same id"
    assert labels[2] != labels[0], "different string -> different id"
    assert sink.objects[0][5] == 0.9


def test_tracking_relates_to_its_detection():
    sink = StubMetaSink()
    analytics.bind(sink)
    meta = analytics.add_relation_meta(buf=None)

    od = analytics.add_object(meta, "person", 1, 2, 3, 4, 0.9)
    track = analytics.add_tracking(meta, 77)
    assert analytics.relate(meta, od, track) is True

    # Handles are the sink's own staging indices, so the relation names the
    # detection and the tracking record that were actually staged.
    assert sink.staged[od][0] == "object"
    assert sink.staged[track] == ("tracking", 77)
    assert sink.relations == [(od, track)]


def test_video_transform_g2g_process_end_to_end():
    """A VideoTransform subclass inverts the frame and stages one detection,
    driven exactly as the host drives it: instance.g2g_process(buf, w, h, fmt, sink)."""

    class InvertAndDetect(VideoTransform):
        def process_frames(self, frames, num_sources, fmt, target):
            assert num_sources == 1
            inverted = 255 - frames
            frameio.write_frame(target, inverted)
            meta = analytics.add_relation_meta(target)
            analytics.add_object(meta, "person", 0, 0, 10, 10, 0.99)

    width, height = 8, 8
    buf = bytearray([10] * (width * height * 3))
    sink = StubMetaSink()

    elem = InvertAndDetect()
    # EngineManager defaults to the pytorch engine, so the first frame would try
    # to load a model. This test covers the frame plumbing, not inference.
    elem.engine_name = None
    ret = elem.g2g_process(buf, width, height, "RGB", sink)

    assert ret is None
    assert all(b == 245 for b in buf), "frame inverted in place (255 - 10)"
    assert len(sink.objects) == 1
    assert sink.objects[0][5] == 0.99
    assert elem.width == width and elem.height == height
