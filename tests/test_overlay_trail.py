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

np = pytest.importorskip("numpy")

from backend import analytics  # noqa: E402

pytestmark = pytest.mark.skipif(
    Gst.ElementFactory.find("mp_overlay") is None,
    reason="the python plugin loader did not register the pyml elements",
)

WIDTH, HEIGHT = 64, 32
BYTES_PER_PIXEL = 4
TRAIL_POINTS = ((8, 16), (56, 16))
BOX = (8, 8, 16, 16)
PIPELINE_TIMEOUT = 10 * Gst.SECOND


def overlaid_frames(labelled_boxes_per_frame):
    pipeline = Gst.parse_launch(
        f"appsrc name=source format=time "
        f"caps=video/x-raw,format=RGBA,width={WIDTH},height={HEIGHT},framerate=30/1 "
        f"! mp_overlay ! appsink name=sink sync=false"
    )
    source = pipeline.get_by_name("source")
    sink = pipeline.get_by_name("sink")
    pipeline.set_state(Gst.State.PLAYING)

    for labelled_boxes in labelled_boxes_per_frame:
        buffer = Gst.Buffer.new_wrapped(bytes(WIDTH * HEIGHT * 4))
        GstVideo.buffer_add_video_meta(
            buffer,
            GstVideo.VideoFrameFlags.NONE,
            GstVideo.VideoFormat.RGBA,
            WIDTH,
            HEIGHT,
        )
        meta = analytics.add_relation_meta(buffer)
        for label, (x, y, w, h) in labelled_boxes:
            analytics.add_object(meta, label, x, y, w, h, 1.0)
        source.emit("push-buffer", buffer)
    source.emit("end-of-stream")

    frames = []
    for _ in labelled_boxes_per_frame:
        sample = sink.emit("try-pull-sample", PIPELINE_TIMEOUT)
        assert sample is not None, "overlay produced no frame"
        frame = np.frombuffer(
            sample.get_buffer().extract_dup(0, WIDTH * HEIGHT * BYTES_PER_PIXEL),
            dtype=np.uint8,
        )
        frames.append(frame.reshape(HEIGHT, WIDTH, BYTES_PER_PIXEL))
    pipeline.set_state(Gst.State.NULL)
    return frames


def test_trail_points_are_joined_into_a_line():
    (frame,) = overlaid_frames(
        [[("stream_0_ball_trail", (x, y, 0, 0)) for x, y in TRAIL_POINTS]]
    )
    midpoint_x = (TRAIL_POINTS[0][0] + TRAIL_POINTS[1][0]) // 2
    red, green, blue, _alpha = frame[TRAIL_POINTS[0][1], midpoint_x]
    assert red > 200 and green > 200 and blue < 50


def test_a_frame_without_detections_gets_no_boxes():
    with_box, without_box = overlaid_frames([[("person", BOX)], []])
    assert with_box.any()
    assert not without_box.any()
