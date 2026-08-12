"""Unit tests for the g2g element backend (`PYML_BACKEND=g2g`).

These exercise the backend with no GStreamer present: the backend selection, the
`GObject` / `FlowReturn` shims, the `G2gFrameIO` buffer round-trip, the
`G2gAnalyticsBackend` mapping onto a flat sink, and an end-to-end `g2g_process`
on a `VideoTransform` subclass. The g2g host's `FrameBuffer` (a writable
buffer-protocol view) and `MetaSink` (write-only staging) are stubbed with a
`bytearray` and a recording object, so no Rust host is needed.

The payload tests at the end drive one text element through both backend
drivers, so they need GStreamer for the gst half and skip without it.
"""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent / "plugins" / "python"

# Select the g2g backend before importing `backend`, and make the plugin package
# importable.
os.environ["PYML_BACKEND"] = "g2g"
sys.path.insert(0, str(PLUGIN_DIR))

import backend  # noqa: E402
from backend import (
    GObject,
    FlowReturn,
    frameio,
    analytics,
    BaseAggregator,
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
        self.class_names = None
        self.emitted = []
        self.emitted_durations = []

    def emit(self, payload, duration_ns=None):
        self.emitted.append(payload)
        self.emitted_durations.append(duration_ns)

    def set_class_names(self, names):
        self.class_names = list(names)

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


#: What the shadowed `gi` raises, so a warning naming it means something wanted it.
NO_PYGOBJECT = "no pygobject"

#: The families that run hosted on g2g, so none of them may need GStreamer.
HOSTED_ELEMENT_MODULES = [
    "base_translate",
    "base_transcribe",
    "base_llm",
    "base_separate",
    "base_tts",
    "base_caption",
    "mariantranslate",
    "whispertranscribe",
    "whisperlive",
    "llm",
    "demucs",
    "sepformer",
    "coquitts",
    "whisperspeechtts",
    "caption_phi",
    "caption_qwen",
]


def test_hosted_elements_import_under_g2g_with_no_pygobject(tmp_path):
    """A fresh interpreter, with `gi` shadowed by one that refuses to import.

    In-process the check cannot be honest: once any test here has initialised
    Gst, these imports succeed whether or not the element guards its Gst
    construction. Shadowing `gi` rather than just checking stderr also makes an
    element that reaches for GStreamer fail here, instead of quietly depending
    on a pygobject that happens to be installed.
    """
    shadow = tmp_path / "gi"
    shadow.mkdir()
    (shadow / "__init__.py").write_text(f'raise ImportError("{NO_PYGOBJECT}")\n')

    result = subprocess.run(
        [sys.executable, "-c", "import " + ", ".join(HOSTED_ELEMENT_MODULES)],
        env={
            **os.environ,
            "PYML_BACKEND": "g2g",
            "PYTHONPATH": os.pathsep.join([str(tmp_path), str(PLUGIN_DIR)]),
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    # The shadow raises with this text, so it appears only if something imported gi.
    assert (
        NO_PYGOBJECT not in result.stderr
    ), "an element reached for GStreamer at import time"


def test_every_element_module_imports_under_g2g_without_gst_init():
    """The same check over the whole plugin directory, so a new element cannot
    quietly reintroduce the crash.

    Kept apart from the payload check above because a module here may warn about
    a dependency it cannot find, which is not what this is looking for. A pad
    template built without `Gst.init` segfaults, so an element that forgets the
    backend guard takes down whatever process imports it.
    """
    pytest.importorskip("gi", reason="the element modules import it at module scope")

    modules = sorted(p.stem for p in PLUGIN_DIR.glob("*.py"))

    result = subprocess.run(
        [sys.executable, "-c", "import " + ", ".join(modules)],
        env={**os.environ, "PYML_BACKEND": "g2g", "PYTHONPATH": str(PLUGIN_DIR)},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


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


def test_a_property_set_from_a_pipeline_line_arrives_as_its_declared_type():
    """The host cannot know a hosted class's property types, so it forwards the
    text a pipeline line carries and the declaration here converts it."""

    class Widget:
        count = GObject.Property(type=int, default=1)
        enabled = GObject.Property(type=bool, default=False)
        ratio = GObject.Property(type=float, default=0.0)
        name = GObject.Property(type=str, default="")

    w = Widget()
    w.count, w.enabled, w.ratio, w.name = "4", "TRUE", "0.5", "3"
    assert (w.count, w.enabled, w.ratio, w.name) == (4, True, 0.5, "3")

    w.enabled = "no"
    assert w.enabled is False
    w.count = 9  # already typed, set from Python
    assert w.count == 9

    with pytest.raises(ValueError, match="enabled"):
        w.enabled = "maybe"


def test_an_element_lists_the_properties_it_declares():
    """The host checks a pipeline against this, so a knob the element has must be
    in it and a name it does not have must not."""
    pytest.importorskip("gi", reason="the video transform needs it on the gst backend")
    pytest.importorskip("torch", reason="the depth engine imports it")
    from depth import DepthTransform

    declared = DepthTransform().g2g_properties()

    assert "colormap" in declared, "the element's own knob"
    assert "batch_size" in declared, "one it inherits from the shared tunables"
    assert "speaker" not in declared, "a detector has no speaker"
    assert len(declared) == len(set(declared)), "an overridden property listed twice"


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


def test_class_names_are_published_so_a_consumer_can_name_a_label():
    sink = StubMetaSink()
    analytics.bind(sink)
    meta = analytics.add_relation_meta(buf=None)

    person = analytics.add_object(meta, "person", 1, 2, 3, 4, 0.9)
    handbag = analytics.add_object(meta, "handbag", 0, 0, 1, 1, 0.5)

    # The table is indexed by the label id staged on the detection, so a
    # consumer holding only the id can look the name up.
    names = sink.class_names
    assert names is not None, "the sink was never sent a name table"
    assert names[sink.staged[person][1]] == "person"
    assert names[sink.staged[handbag][1]] == "handbag"


def test_class_names_are_resent_for_each_frames_sink():
    """Each frame gets a fresh sink, so a name interned on an earlier frame has
    to be published again rather than assumed already known."""
    first = StubMetaSink()
    analytics.bind(first)
    staged = analytics.add_object(
        analytics.add_relation_meta(None), "person", 1, 2, 3, 4, 0.9
    )
    assert first.class_names[first.staged[staged][1]] == "person"

    second = StubMetaSink()
    analytics.bind(second)
    staged = analytics.add_object(
        analytics.add_relation_meta(None), "person", 5, 6, 7, 8, 0.8
    )
    assert second.class_names is not None, "the second sink was sent no table"
    assert second.class_names[second.staged[staged][1]] == "person"


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


def test_aggregator_drives_the_same_hook_a_transform_fills_in():
    """The N-source case spells the same on both backends: one `process_frames`
    taking (H, W, C) for a single source and (N, H, W, C) for several."""
    seen = []

    class Batching(BaseAggregator):
        def process_frames(self, frames, num_sources, fmt, target):
            seen.append((frames.shape, num_sources, fmt))

    width, height = 4, 3
    elem = Batching()
    elem.engine_name = None

    buffers = [bytearray([n] * (width * height * 3)) for n in (1, 2)]
    elem.g2g_process_batch(buffers, width, height, "RGB", StubMetaSink())
    elem.g2g_process_batch(buffers[:1], width, height, "RGB", StubMetaSink())

    assert seen == [
        ((2, height, width, 3), 2, "RGB"),
        ((height, width, 3), 1, "RGB"),
    ]


def test_g2g_caption_stages_the_caption_on_the_frame():
    """The caption family runs on the shared per-frame seam, so it works with no
    text pad and no GStreamer: the caption is staged as a classification."""
    from base_caption import BaseCaption

    class FakeCaption(BaseCaption):
        def forward(self, frames):
            return "a cat on a mat"

    leaf = FakeCaption()
    leaf.mgr.engine_name = None  # the fake above is the model
    sink = StubMetaSink()
    width, height = 4, 3

    leaf.g2g_process(bytearray(width * height * 4), width, height, "RGBA", sink)

    captions = [record for record in sink.staged if record[0] == "classification"]
    assert len(captions) == 1, "the caption was not staged"
    assert sink.class_names[captions[0][1]] == "a cat on a mat"


def test_a_packed_frame_is_reduced_to_rgb_in_channel_order():
    from backend.g2g.frameio import as_rgb

    pixel = np.array([[[10, 20, 30, 40]]], dtype=np.uint8)

    assert as_rgb(pixel, "RGBA").tolist() == [[[10, 20, 30]]]
    assert as_rgb(pixel, "BGRA").tolist() == [[[30, 20, 10]]]
    assert as_rgb(pixel, "ARGB").tolist() == [[[20, 30, 40]]]
    assert as_rgb(pixel, "ABGR").tolist() == [[[40, 30, 20]]]

    rgb = np.array([[[10, 20, 30]]], dtype=np.uint8)
    assert as_rgb(rgb, "RGB") is rgb, "an RGB frame is handed over untouched"
    assert as_rgb(rgb, "BGR").tolist() == [[[30, 20, 10]]]


def test_two_elements_on_their_own_threads_keep_their_own_sinks():
    """The host runs one thread per element, so a binding made on one thread must
    not redirect what another thread stages."""
    import threading

    sinks = {}
    staged = {}
    start = threading.Barrier(2)
    bound = threading.Barrier(2)

    def stage(name, label):
        sinks[name] = StubMetaSink()
        start.wait()
        analytics.bind(sinks[name])
        frameio.bind(sinks[name], "RGB")
        # Both threads have bound before either stages anything, so a shared
        # binding would send both records to whichever bound last.
        bound.wait()
        analytics.add_object(analytics.add_relation_meta(None), label, 0, 0, 1, 1, 1.0)
        frameio.append_blob(bytearray(4), "tag", label.encode())
        staged[name] = sinks[name].staged

    threads = [
        threading.Thread(target=stage, args=(name, label))
        for name, label in (("first", "person"), ("second", "handbag"))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(staged["first"]) == 2, "the first element lost a record to the second"
    assert len(staged["second"]) == 2
    assert sinks["first"].blobs == [("tag", b"person")]
    assert sinks["second"].blobs == [("tag", b"handbag")]


class RecordingLogger:
    """Stands in for the element's logger, keeping what it was told."""

    def __init__(self):
        self.warnings = []

    def warning(self, message):
        self.warnings.append(message)

    def info(self, message):
        pass

    def error(self, message):
        pass


def gst():
    pytest.importorskip("gi", reason="the gst driver and the pad templates need it")
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    return Gst


def gst_payload_driver():
    """The gst backend's driver half, to mix into a leaf.

    This process selected the g2g backend, so a leaf's element base is the g2g
    one. Adding the driver gives a single instance both backends' entry points,
    which is what the seam claims: the same element runs under either.
    """
    gst()
    from backend.gst.aggregator import PayloadDriver

    return PayloadDriver


CHUNKED_OUTPUTS = [b"one", b"two", b"three"]


def chunking_leaf():
    """A payload element that answers one input buffer with several outputs."""
    driver = gst_payload_driver()

    class ChunkingLeaf(BaseAggregator, driver):
        def process_payload(self, payload):
            return list(CHUNKED_OUTPUTS)

    leaf = ChunkingLeaf()
    leaf.engine_name = None
    leaf.logger = RecordingLogger()
    return leaf


def translate_leaf(translate_text):
    """A real `BaseTranslate` whose model is the given text -> text function.

    Importing the element needs GStreamer even under the g2g backend, since the
    family still declares its pad templates with Gst types.
    """
    driver = gst_payload_driver()
    from base_translate import BaseTranslate

    class FakeTranslate(BaseTranslate, driver):
        def do_translate_text(self, text):
            return translate_text(text)

    leaf = FakeTranslate()
    leaf.engine_name = None  # nothing to load: the fake above is the model
    return leaf


class StubSrcPad:
    """Stands in for the element's src pad, for the family that pushes straight
    out of it instead of through the aggregator."""

    def __init__(self, pushed):
        self._pushed = pushed
        self.push_count = 0

    def push(self, buf):
        self._pushed.append(buf)
        self.push_count += 1
        return 0


def drive_gst_payload(leaf, payload, pts=1000, duration=500):
    """Run the gst driver over one input buffer, returning what it sent.

    Both send routes are stubbed, the aggregator's `finish_buffer` and the src
    pad, so `leaf.srcpad.push_count` says which one the element took.
    """
    Gst = gst()
    from backend.gst.aggregator import BaseAggregator as GstBaseAggregator

    pushed = []
    leaf.finish_buffer = pushed.append
    leaf.srcpad = StubSrcPad(pushed)

    inbuf = Gst.Buffer.new_allocate(None, len(payload), None)
    inbuf.fill(0, payload)
    inbuf.pts = pts
    inbuf.duration = duration

    ret = GstBaseAggregator.do_process(leaf, inbuf)
    return ret, pushed


def buffer_bytes(buf):
    Gst = gst()
    success, map_info = buf.map(Gst.MapFlags.READ)
    assert success
    data = bytes(map_info.data)
    buf.unmap(map_info)
    return data


def test_g2g_payload_driver_emits_the_translated_bytes():
    leaf = translate_leaf(lambda text: "hello" if text == "hola" else "")
    sink = StubMetaSink()

    leaf.g2g_process_payload([bytearray(b"hola")], "text/x-raw,format=utf8", sink)

    assert sink.emitted == [b"hello"]
    assert sink.emitted_durations == [None], "text keeps the input buffer's timing"


def test_g2g_payload_driver_emits_nothing_when_the_element_has_no_output():
    leaf = translate_leaf(lambda text: "")
    sink = StubMetaSink()

    leaf.g2g_process_payload([bytearray(b"hola")], "text/x-raw,format=utf8", sink)

    assert sink.emitted == [], "an empty result must not reach the host"


def test_g2g_payload_driver_emits_every_payload():
    leaf = chunking_leaf()
    sink = StubMetaSink()

    leaf.g2g_process_payload([bytearray(b"in")], "audio/x-raw", sink)

    assert sink.emitted == CHUNKED_OUTPUTS
    assert leaf.logger.warnings == []


def test_gst_payload_driver_pushes_the_translated_buffer():
    Gst = gst()
    leaf = translate_leaf(lambda text: "hello" if text == "hola" else "")

    ret, pushed = drive_gst_payload(leaf, b"hola")

    assert ret == Gst.FlowReturn.OK
    assert len(pushed) == 1
    assert buffer_bytes(pushed[0]) == b"hello"
    assert pushed[0].pts == 1000 and pushed[0].duration == 500


def test_gst_payload_driver_pushes_nothing_when_the_element_has_no_output():
    Gst = gst()
    leaf = translate_leaf(lambda text: "")

    ret, pushed = drive_gst_payload(leaf, b"hola")

    assert ret == Gst.FlowReturn.OK
    assert pushed == []


def test_gst_payload_driver_pushes_every_payload_as_its_own_buffer():
    leaf = chunking_leaf()

    _, pushed = drive_gst_payload(leaf, b"in")

    assert [buffer_bytes(buf) for buf in pushed] == CHUNKED_OUTPUTS
    assert leaf.logger.warnings == []


VAD_CHUNK_SAMPLES = 2400  # 150 ms at 16 kHz, so two silent chunks end a clip


class Segment:
    """One piece of a transcript, as the Whisper models hand it back."""

    def __init__(self, text):
        self.text = text


def stub_vad(monkeypatch):
    """Stand in for the optional VAD package the transcribe family builds in its
    constructor. This one calls any non-zero sample speech, which lets a test
    write silence and speech as buffer contents."""
    import types

    class SpeechIsNonZero:
        def chunk_samples(self):
            return VAD_CHUNK_SAMPLES

        def process_chunk(self, chunk):
            return 1.0 if any(chunk) else 0.0

    module = types.ModuleType("pysilero_vad")
    module.SileroVoiceActivityDetector = SpeechIsNonZero
    monkeypatch.setitem(sys.modules, "pysilero_vad", module)


def transcribe_leaf(monkeypatch, transcript):
    """A real `BaseTranscribe` with a scripted VAD and transcriber."""
    driver = gst_payload_driver()
    stub_vad(monkeypatch)

    from base_transcribe import BaseTranscribe

    class FakeTranscribe(BaseTranscribe, driver):
        def do_transcribe(self, audio_data, task):
            return [Segment(word) for word in transcript.split()]

    leaf = FakeTranscribe()
    leaf.engine_name = None
    return leaf


def speech(chunks=1):
    return np.full(VAD_CHUNK_SAMPLES * chunks, 1000, dtype=np.int16).tobytes()


def silence(chunks=1):
    return np.zeros(VAD_CHUNK_SAMPLES * chunks, dtype=np.int16).tobytes()


def separate_leaf(sample_rate=4):
    """A real `BaseSeparate` whose model returns the audio it was given."""
    driver = gst_payload_driver()
    from base_separate import BaseSeparate

    class PassThroughSeparate(BaseSeparate, driver):
        SAMPLE_RATE = sample_rate

        def do_separate(self, audio_data):
            return audio_data

    leaf = PassThroughSeparate()
    leaf.engine_name = None
    leaf.streaming = True  # a one second chunk, so four samples at this rate
    return leaf


class FakeEngine:
    def do_generate(self, text, system_prompt=None):
        return "answered " + text


def llm_leaf():
    """A real `BaseLlm` whose engine is a canned generator."""
    driver = gst_payload_driver()
    from base_llm import BaseLlm

    class FakeLlm(BaseLlm, driver):
        @property
        def engine(self):
            return self._fake_engine

        def get_tokenizer(self):
            return "tokenizer"

        def get_model(self):
            return "model"

    leaf = FakeLlm()
    leaf._fake_engine = FakeEngine()
    leaf.engine_name = None
    return leaf


PAYLOAD_FAMILY_BASES = [
    ("base_translate", "BaseTranslate"),
    ("base_transcribe", "BaseTranscribe"),
    ("base_llm", "BaseLlm"),
    ("base_separate", "BaseSeparate"),
    ("base_tts", "BaseTts"),
]


def declared_properties(cls):
    """Every property the class declares or inherits, mapped to its owner."""
    return {
        name: klass.__name__
        for klass in reversed(cls.__mro__)
        for name, value in vars(klass).items()
        if isinstance(value, GObject.Property)
    }


@pytest.mark.parametrize("module_name,class_name", PAYLOAD_FAMILY_BASES)
def test_every_declared_property_reads_before_anything_sets_it(
    monkeypatch, module_name, class_name
):
    """A freshly built element has to answer every property it declares.

    Nothing applies a declared default: GObject keeps it in the pspec and never
    routes it through the setter, so the constructor is the only thing that can
    create the backing attribute a custom getter reads.
    """
    gst()
    stub_vad(monkeypatch)
    element = getattr(importlib.import_module(module_name), class_name)()

    unreadable = []
    for name, owner in sorted(declared_properties(type(element)).items()):
        try:
            getattr(element, name)
        except AttributeError as exception:
            unreadable.append(f"{owner}.{name}: {exception}")

    assert unreadable == []


def test_g2g_transcribe_emits_only_once_the_clip_ends(monkeypatch):
    leaf = transcribe_leaf(monkeypatch, "hello world")
    sink = StubMetaSink()
    caps = "audio/x-raw,format=S16LE,rate=16000,channels=1"

    leaf.g2g_process_payload([bytearray(speech())], caps, sink)
    assert sink.emitted == [], "speech is still being accumulated"

    leaf.g2g_process_payload([bytearray(silence(3))], caps, sink)
    assert sink.emitted == [b"hello world"], "the silence should end the clip"


def test_gst_transcribe_pushes_only_once_the_clip_ends(monkeypatch):
    leaf = transcribe_leaf(monkeypatch, "hello world")

    _, pushed = drive_gst_payload(leaf, speech())
    assert pushed == [], "speech is still being accumulated"

    _, pushed = drive_gst_payload(leaf, silence(3))
    assert [buffer_bytes(buf) for buf in pushed] == [b"hello world"]
    assert pushed[0].pts == 1000 and pushed[0].duration == 500


def test_gst_transcribe_pushes_nothing_for_a_buffer_below_one_vad_chunk(monkeypatch):
    leaf = transcribe_leaf(monkeypatch, "hello world")

    _, pushed = drive_gst_payload(leaf, np.zeros(8, dtype=np.int16).tobytes())

    assert pushed == []


def test_gst_separate_pushes_one_buffer_per_whole_chunk():
    leaf = separate_leaf()
    samples = np.arange(1, 11, dtype=np.int16)  # ten samples, so two chunks of four

    _, pushed = drive_gst_payload(leaf, samples.tobytes())

    assert [buffer_bytes(buf) for buf in pushed] == [
        samples[:4].tobytes(),
        samples[4:8].tobytes(),
    ]
    assert len(leaf.clip_buffer) == 2, "the remainder waits for the next buffer"


def test_g2g_separate_emits_one_buffer_per_whole_chunk():
    leaf = separate_leaf()
    leaf.logger = RecordingLogger()
    sink = StubMetaSink()
    samples = np.arange(1, 11, dtype=np.int16)

    leaf.g2g_process_payload([bytearray(samples.tobytes())], "audio/x-raw", sink)

    assert sink.emitted == [samples[:4].tobytes(), samples[4:8].tobytes()]
    assert leaf.logger.warnings == []


def test_g2g_llm_emits_the_generated_text():
    leaf = llm_leaf()
    sink = StubMetaSink()

    leaf.g2g_process_payload([bytearray(b"question")], "text/x-raw,format=utf8", sink)

    assert sink.emitted == [b"answered question"]


def test_gst_llm_pushes_out_of_the_src_pad_rather_than_the_aggregator():
    Gst = gst()
    leaf = llm_leaf()

    ret, pushed = drive_gst_payload(leaf, b"question")

    assert ret == Gst.FlowReturn.OK
    assert [buffer_bytes(buf) for buf in pushed] == [b"answered question"]
    assert leaf.srcpad.push_count == 1, "this family has never used finish_buffer"
    assert pushed[0].pts == 1000 and pushed[0].duration == 500


TTS_SAMPLE_RATE = 22050
TTS_SAMPLES = 8000


def tts_leaf():
    """A real `BaseTts` whose voice is a ramp of the right sample count."""
    driver = gst_payload_driver()
    from base_tts import BaseTts

    class FakeTts(BaseTts, driver):
        def do_load_model(self):
            pass

        def do_generate_speech(self, transcript):
            return np.linspace(-0.5, 0.5, TTS_SAMPLES, dtype=np.float32)

        def do_get_sample_rate(self):
            return TTS_SAMPLE_RATE

    leaf = FakeTts()
    leaf.engine_name = None
    return leaf


def expected_tts_duration_ns():
    return int(TTS_SAMPLES / TTS_SAMPLE_RATE * 1_000_000_000)


def test_gst_tts_stamps_its_own_duration_and_no_presentation_time():
    Gst = gst()
    leaf = tts_leaf()

    ret, pushed = drive_gst_payload(leaf, b"speak this")

    assert ret == Gst.FlowReturn.OK
    assert len(pushed) == 1
    assert len(buffer_bytes(pushed[0])) == TTS_SAMPLES * 2, "S16LE, one channel"
    assert (
        pushed[0].pts == Gst.CLOCK_TIME_NONE
    ), "the text buffer's pts is not the audio's"
    assert pushed[0].dts == Gst.CLOCK_TIME_NONE
    assert pushed[0].duration == expected_tts_duration_ns()
    assert leaf.srcpad.push_count == 1, "this family has never used finish_buffer"


def test_g2g_tts_emits_the_duration_the_audio_actually_runs_for():
    leaf = tts_leaf()
    sink = StubMetaSink()

    leaf.g2g_process_payload([bytearray(b"speak this")], "text/x-raw,format=utf8", sink)

    assert len(sink.emitted) == 1
    assert len(sink.emitted[0]) == TTS_SAMPLES * 2
    assert sink.emitted_durations == [expected_tts_duration_ns()]


def test_tts_streaming_speaks_each_chunk_of_text_separately():
    """Streaming splits the text into 20 character chunks, one payload each."""
    gst()
    leaf = tts_leaf()
    leaf.streaming = True

    _, pushed = drive_gst_payload(leaf, b"x" * 45)

    assert len(pushed) == 3, "45 characters is three chunks"
    assert {buf.duration for buf in pushed} == {expected_tts_duration_ns()}


def test_g2g_tts_streaming_emits_each_chunk_of_text_separately():
    leaf = tts_leaf()
    leaf.streaming = True
    sink = StubMetaSink()

    leaf.g2g_process_payload([bytearray(b"x" * 45)], "text/x-raw,format=utf8", sink)

    assert len(sink.emitted) == 3, "45 characters is three chunks"
    assert sink.emitted_durations == [expected_tts_duration_ns()] * 3


def test_gst_driver_keeps_the_input_timing_including_dts():
    """base_llm timestamped its output with the input's dts; the driver, which
    now builds that buffer, has to keep doing it."""
    Gst = gst()
    leaf = llm_leaf()

    pushed = []
    leaf.finish_buffer = pushed.append
    leaf.srcpad = StubSrcPad(pushed)
    inbuf = Gst.Buffer.new_allocate(None, len(b"question"), None)
    inbuf.fill(0, b"question")
    inbuf.pts = 90
    inbuf.dts = 80
    inbuf.duration = 70

    from backend.gst.aggregator import BaseAggregator as GstBaseAggregator

    GstBaseAggregator.do_process(leaf, inbuf)

    assert (pushed[0].pts, pushed[0].dts, pushed[0].duration) == (90, 80, 70)
