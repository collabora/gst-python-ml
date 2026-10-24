import json
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
gi.require_version("GstVideo", "1.0")
from gi.repository import Gst, GstVideo  # noqa: E402

Gst.init(None)

pytest.importorskip("cv2")

from backend import analytics  # noqa: E402
from utils.blobs import read_blobs  # noqa: E402

pytestmark = pytest.mark.skipif(
    Gst.ElementFactory.find("mp_football_overlay") is None,
    reason="the python plugin loader did not register the pyml elements",
)

WIDTH, HEIGHT = 320, 240
FRAMERATE = 30
PLAYER_ID = 3
PLAYER_BOX = (100, 80, 40, 120)
BALL_SIZE = 10
BALL_START = (115, 185)
STILL_FRAMES = 4
FLIGHT_FRAMES = 4
# two ball sizes a frame
BALL_STEP = 2 * BALL_SIZE
PIPELINE_TIMEOUT = 10 * Gst.SECOND


def ball_positions():
    x, y = BALL_START
    still = [(x, y)] * STILL_FRAMES
    flight = [(x + BALL_STEP * step, y) for step in range(1, FLIGHT_FRAMES + 1)]
    return still + flight


def frame_buffer(index, ball):
    buffer = Gst.Buffer.new_wrapped(bytes(WIDTH * HEIGHT * 4))
    buffer.pts = index * Gst.SECOND // FRAMERATE
    buffer.duration = Gst.SECOND // FRAMERATE
    GstVideo.buffer_add_video_meta(
        buffer, GstVideo.VideoFrameFlags.NONE, GstVideo.VideoFormat.RGBA, WIDTH, HEIGHT
    )
    meta = analytics.add_relation_meta(buffer)
    analytics.add_object(meta, f"player_id_{PLAYER_ID}", *PLAYER_BOX, 1.0)
    analytics.add_object(meta, "ball_id_9", *ball, BALL_SIZE, BALL_SIZE, 1.0)
    return buffer


def test_a_kick_puts_an_alert_blob_on_its_frame_and_counts_once():
    pipeline = Gst.parse_launch(
        f"appsrc name=source format=time "
        f"caps=video/x-raw,format=RGBA,width={WIDTH},height={HEIGHT},"
        f"framerate={FRAMERATE}/1 "
        "! mp_football_overlay name=overlay detect-kicks=true team-colors=false "
        "! appsink name=sink sync=false"
    )
    source = pipeline.get_by_name("source")
    sink = pipeline.get_by_name("sink")
    overlay = pipeline.get_by_name("overlay")
    pipeline.set_state(Gst.State.PLAYING)
    for index, ball in enumerate(ball_positions()):
        source.emit("push-buffer", frame_buffer(index, ball))
    source.emit("end-of-stream")

    alerts = []
    for _ in ball_positions():
        sample = sink.emit("try-pull-sample", PIPELINE_TIMEOUT)
        assert sample is not None, "overlay produced too few frames"
        blobs = read_blobs(sample.get_buffer())
        if "alert" in blobs:
            alerts.append(json.loads(blobs["alert"]))
    kick_count = overlay.get_property("kick-count")
    last_kick = overlay.get_property("last-kick")
    pipeline.set_state(Gst.State.NULL)

    assert len(alerts) == 1
    assert alerts[0][0]["event"] == "kick"
    assert alerts[0][0]["player"] == PLAYER_ID
    assert kick_count == 1
    assert json.loads(last_kick)["player"] == PLAYER_ID
