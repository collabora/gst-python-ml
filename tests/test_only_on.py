import os
import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
PLUGIN_DIR = BASE_DIR / "plugins" / "python"

# the plugin loader reads GST_PLUGIN_PATH at Gst.init
os.environ["GST_PLUGIN_PATH"] = str(BASE_DIR / "plugins")
sys.path.insert(0, str(PLUGIN_DIR))

gi = pytest.importorskip("gi")
gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
from gi.repository import GObject, Gst, GstBase  # noqa: E402

Gst.init(None)

from backend import analytics, frameio, VideoTransform  # noqa: E402

pytestmark = pytest.mark.skipif(
    Gst.ElementFactory.find("pyml_alert") is None,
    reason="the python plugin loader did not register the pyml elements",
)

FRAMERATE = 30
TAGGED_FRAME = 4
FRAMES = 10
PIPELINE_TIMEOUT = 30 * Gst.SECOND
ALERT_HEADER = b"GST-ALERT:"
ALERT_PAYLOAD = b'[{"detection": {"label": "thing"}}]'


def run_to_eos(pipeline):
    pipeline.set_state(Gst.State.PLAYING)
    message = pipeline.get_bus().timed_pop_filtered(
        PIPELINE_TIMEOUT, Gst.MessageType.EOS | Gst.MessageType.ERROR
    )
    pipeline.set_state(Gst.State.NULL)
    assert message is not None, "pipeline did not finish"
    if message.type == Gst.MessageType.ERROR:
        error, debug = message.parse_error()
        pytest.fail(f"{error.message}\n{debug}")


# a pad probe's buffer is not writable, an in-place transform's is
class TagOneFrame(GstBase.BaseTransform):
    __gstmetadata__ = (
        "Tag One Frame",
        "Filter",
        "adds an alert blob and a detection to one frame",
        "test",
    )
    __gsttemplates__ = (
        Gst.PadTemplate.new(
            "src", Gst.PadDirection.SRC, Gst.PadPresence.ALWAYS, Gst.Caps.new_any()
        ),
        Gst.PadTemplate.new(
            "sink", Gst.PadDirection.SINK, Gst.PadPresence.ALWAYS, Gst.Caps.new_any()
        ),
    )

    def __init__(self):
        super().__init__()
        self.set_in_place(True)

    def do_transform_ip(self, buffer):
        if buffer.pts == TAGGED_FRAME * Gst.SECOND // FRAMERATE:
            meta = analytics.add_relation_meta(buffer)
            analytics.add_object(meta, "thing", 0, 0, 10, 10, 0.9)
            frameio.append_blob(buffer, ALERT_HEADER, ALERT_PAYLOAD)
        return Gst.FlowReturn.OK


# no engine_name, so nothing loads a model and process_frames stands in for one
class CountFrames(VideoTransform):
    __gstmetadata__ = (
        "Count Frames",
        "Transform",
        "counts the frames that reach process_frames",
        "test",
    )
    # pygobject reads the templates off the class's own dict, not the base's
    __gsttemplates__ = VideoTransform.__gsttemplates__

    def __init__(self):
        super().__init__()
        self.processed = 0

    def process_frames(self, frames, num_sources, fmt, target):
        self.processed += 1


GObject.type_register(TagOneFrame)
Gst.Element.register(None, "tagoneframe", Gst.Rank.NONE, TagOneFrame)
GObject.type_register(CountFrames)
Gst.Element.register(None, "countframes", Gst.Rank.NONE, CountFrames)


def processed_frames(only_on=None):
    gating = f"only-on={only_on} " if only_on else ""
    pipeline = Gst.parse_launch(
        f"videotestsrc num-buffers={FRAMES} "
        f"! video/x-raw,width=64,height=48,framerate={FRAMERATE}/1,format=RGBA "
        "! tagoneframe "
        f"! countframes name=counter {gating}"
        "! fakesink"
    )
    counter = pipeline.get_by_name("counter")
    run_to_eos(pipeline)
    return counter.processed


def test_every_frame_runs_when_only_on_is_unset():
    assert processed_frames() == FRAMES


def test_only_the_frame_carrying_the_alert_blob_runs():
    assert processed_frames("alert") == 1


def test_only_the_frame_carrying_a_detection_runs():
    assert processed_frames("detections") == 1


def test_a_blob_nobody_attaches_stops_every_frame():
    assert processed_frames("caption") == 0
