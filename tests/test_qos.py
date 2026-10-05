import os
import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
PLUGIN_DIR = BASE_DIR / "plugins" / "python"

os.environ["GST_PLUGIN_PATH"] = str(BASE_DIR / "plugins")
sys.path.insert(0, str(PLUGIN_DIR))

gi = pytest.importorskip("gi")
gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
from gi.repository import GObject, Gst  # noqa: E402

Gst.init(None)

from backend import VideoTransform  # noqa: E402

FRAMERATE = 30
FRAMES = 90
PIPELINE_TIMEOUT = 30 * Gst.SECOND
# each frame costs two frame periods
SLOW_FRAME_SECONDS = 2 / FRAMERATE


# no engine_name, so nothing loads a model
class SlowFrames(VideoTransform):
    __gstmetadata__ = (
        "Slow Frames",
        "Transform",
        "sleeps two frame periods per frame and counts the frames it saw",
        "test",
    )
    __gsttemplates__ = VideoTransform.__gsttemplates__

    def __init__(self):
        super().__init__()
        self.processed = 0

    def process_frames(self, frames, num_sources, fmt, target):
        import time

        self.processed += 1
        time.sleep(SLOW_FRAME_SECONDS)


GObject.type_register(SlowFrames)
Gst.Element.register(None, "slowframes", Gst.Rank.NONE, SlowFrames)


def run_to_eos(pipeline):
    pipeline.set_state(Gst.State.PLAYING)
    bus = pipeline.get_bus()
    sink_dropped = 0
    while True:
        message = bus.timed_pop_filtered(
            PIPELINE_TIMEOUT,
            Gst.MessageType.EOS | Gst.MessageType.ERROR | Gst.MessageType.QOS,
        )
        assert message is not None, "pipeline did not finish"
        if message.type == Gst.MessageType.QOS:
            _, _, dropped = message.parse_qos_stats()
            sink_dropped = max(sink_dropped, dropped)
            continue
        break
    pipeline.set_state(Gst.State.NULL)
    if message.type == Gst.MessageType.ERROR:
        error, debug = message.parse_error()
        pytest.fail(f"{error.message}\n{debug}")
    return sink_dropped


# non-live source: frames arrive early, so the element can skip ahead
def run_slow_pipeline(qos):
    pipeline = Gst.parse_launch(
        f"videotestsrc num-buffers={FRAMES} "
        f"! video/x-raw,width=64,height=48,framerate={FRAMERATE}/1,format=RGBA "
        f"! slowframes name=slow qos={str(qos).lower()} "
        "! fakesink sync=true qos=true max-lateness=20000000"
    )
    slow = pipeline.get_by_name("slow")
    sink_dropped = run_to_eos(pipeline)
    return slow.processed, sink_dropped


def test_qos_is_on_by_default():
    assert Gst.ElementFactory.make("slowframes").is_qos_enabled()


def test_without_qos_every_frame_runs_and_the_sink_drops_them():
    processed, sink_dropped = run_slow_pipeline(qos=False)
    assert processed == FRAMES
    assert sink_dropped > FRAMES // 2


def test_a_slow_element_skips_to_frames_the_sink_can_still_show():
    processed, sink_dropped = run_slow_pipeline(qos=True)
    assert processed < FRAMES
    assert sink_dropped * 4 < processed, (processed, sink_dropped)
