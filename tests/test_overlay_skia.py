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
pytest.importorskip("skia")

from backend import analytics  # noqa: E402

pytestmark = pytest.mark.skipif(
    Gst.ElementFactory.find("pyml_overlay") is None,
    reason="the python plugin loader did not register the pyml elements",
)

WIDTH, HEIGHT = 64, 32
TRAIL_POINTS = ((8, 16), (56, 16))
PERSON_BOX = (8, 4, 40, 20)
PIPELINE_TIMEOUT = 10 * Gst.SECOND


def overlaid_frame(labelled_boxes):
    pipeline = Gst.parse_launch(
        f"appsrc name=source format=time "
        f"caps=video/x-raw,format=RGBA,width={WIDTH},height={HEIGHT},framerate=30/1 "
        f"! pyml_overlay renderer=skia ! appsink name=sink sync=false"
    )
    source = pipeline.get_by_name("source")
    sink = pipeline.get_by_name("sink")
    pipeline.set_state(Gst.State.PLAYING)

    buffer = Gst.Buffer.new_wrapped(bytes(WIDTH * HEIGHT * 4))
    GstVideo.buffer_add_video_meta(
        buffer, GstVideo.VideoFrameFlags.NONE, GstVideo.VideoFormat.RGBA, WIDTH, HEIGHT
    )
    meta = analytics.add_relation_meta(buffer)
    for label, (x, y, w, h) in labelled_boxes:
        analytics.add_object(meta, label, x, y, w, h, 1.0)
    source.emit("push-buffer", buffer)
    source.emit("end-of-stream")

    sample = sink.emit("try-pull-sample", PIPELINE_TIMEOUT)
    pipeline.set_state(Gst.State.NULL)
    assert sample is not None, "overlay produced no frame"
    output = sample.get_buffer()
    frame = np.frombuffer(output.extract_dup(0, output.get_size()), dtype=np.uint8)
    return frame.reshape(HEIGHT, WIDTH, 4)


def test_trail_points_are_joined_into_a_line():
    frame = overlaid_frame(
        [("stream_0_ball_trail", (x, y, 0, 0)) for x, y in TRAIL_POINTS]
    )
    midpoint_x = (TRAIL_POINTS[0][0] + TRAIL_POINTS[1][0]) // 2
    red, green, blue, _alpha = frame[TRAIL_POINTS[0][1], midpoint_x]
    assert red > 200 and green > 200 and blue < 50


def test_box_is_stroked_in_red():
    frame = overlaid_frame([("stream_0_person", PERSON_BOX)])
    box_x, box_y, box_width, _box_height = PERSON_BOX
    red, green, blue, _alpha = frame[box_y, box_x + box_width // 2]
    assert red > 200 and green < 50 and blue < 50
