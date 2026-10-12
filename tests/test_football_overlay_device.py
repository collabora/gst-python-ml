import os
import sys
from pathlib import Path

import numpy as np
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent

# the plugin loader reads GST_PLUGIN_PATH at Gst.init
os.environ["GST_PLUGIN_PATH"] = str(BASE_DIR / "plugins")
sys.path.insert(0, str(BASE_DIR / "plugins" / "python"))

gi = pytest.importorskip("gi")
gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import Gst, GstVideo  # noqa: E402

Gst.init(None)
torch = pytest.importorskip("torch")
pytest.importorskip("cv2")

from backend import analytics  # noqa: E402
from test_device_write import device_pixels, on_a_streaming_thread  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and Gst.ElementFactory.find("cudaupload")),
    reason="needs a cuda device and the cudaupload element",
)

WIDTH, HEIGHT = 1280, 720
RGBA_CHANNELS = 4
RGB_CHANNELS = 3
SYSTEM_CAPS = f"video/x-raw,format=RGBA,width={WIDTH},height={HEIGHT}"
DEVICE_CAPS = (
    f"video/x-raw(memory:CUDAMemory),format=RGBA,width={WIDTH},height={HEIGHT}"
)
PULL_TIMEOUT = 10 * Gst.SECOND
PITCH_RGBA = (40, 140, 40, 255)
JERSEY_RGBA = (255, 0, 0, 255)
RED_TEAM_RGB = (255, 40, 40)
HUD_BACKGROUND_RGB = (92, 41, 131)
PLAYER_LABEL = "player_id_3"
BALL_LABEL = "ball_id_9"
# (x, y, width, height)
PLAYER_BOX = (600, 300, 60, 150)
BALL_BOX = (800, 500, 12, 12)
# the torso window the jersey vote samples
JERSEY_TOP, JERSEY_BOTTOM = 0.15, 0.55
JERSEY_LEFT, JERSEY_RIGHT = 0.25, 0.75
# a team colour needs four jersey votes
FRAMES_UNTIL_TEAM_COLOUR = 4
ELLIPSE_HEIGHT_RATIO = 0.35
# covers the hud with the default headshot and its border
HUD_REGION = (slice(0, 130), slice(0, 400))
# the device draws with other fonts and without antialiasing
DRAWN_PIXEL_MISMATCH_FRACTION = 0.003
OVERLAY_PROPERTIES = {
    "team-colors": True,
    "show-ball": True,
    "trails": False,
    "show-hud": True,
    "highlight-focal": False,
    "show-ids": False,
    "show-labels": False,
    "headshot-path": str(BASE_DIR / "data" / "Chinedu-Obasi_2684938.jpg"),
}


def pitch_frame():
    frame = np.empty((HEIGHT, WIDTH, RGBA_CHANNELS), np.uint8)
    frame[:] = PITCH_RGBA
    x, y, width, height = PLAYER_BOX
    frame[y : y + height, x : x + width] = (250, 250, 250, 255)
    top, bottom = y + int(JERSEY_TOP * height), y + int(JERSEY_BOTTOM * height)
    left, right = x + int(JERSEY_LEFT * width), x + int(JERSEY_RIGHT * width)
    frame[top:bottom, left:right] = JERSEY_RGBA
    x, y, width, height = BALL_BOX
    frame[y : y + height, x : x + width] = (255, 255, 255, 255)
    return frame


def system_buffer(frame):
    buffer = Gst.Buffer.new_allocate(None, frame.nbytes, None)
    buffer.fill(0, frame.tobytes())
    GstVideo.buffer_add_video_meta(
        buffer, GstVideo.VideoFrameFlags.NONE, GstVideo.VideoFormat.RGBA, WIDTH, HEIGHT
    )
    return buffer


def uploaded_sample(frame):
    pipeline = Gst.parse_launch(
        f"appsrc name=source format=time caps={SYSTEM_CAPS},framerate=30/1 "
        f"! cudaupload ! {DEVICE_CAPS} ! appsink name=sink sync=false"
    )
    pipeline.set_state(Gst.State.PLAYING)
    try:
        pipeline.get_by_name("source").emit("push-buffer", system_buffer(frame))
        sample = pipeline.get_by_name("sink").emit("try-pull-sample", PULL_TIMEOUT)
        assert sample is not None
    finally:
        pipeline.set_state(Gst.State.NULL)
    return sample


def with_analytics(buffer):
    meta = analytics.add_relation_meta(buffer)
    analytics.add_object(meta, PLAYER_LABEL, *PLAYER_BOX, 1.0)
    analytics.add_object(meta, BALL_LABEL, *BALL_BOX, 1.0)
    return buffer


def football_overlay(caps):
    from football_overlay import FootballOverlay

    overlay = FootballOverlay()
    for name, value in OVERLAY_PROPERTIES.items():
        overlay.set_property(name, value)
    parsed_caps = Gst.Caps.from_string(caps)
    overlay.do_set_caps(parsed_caps, parsed_caps)
    return overlay


def draw_system_frames(frame):
    overlay = football_overlay(SYSTEM_CAPS)
    for _ in range(FRAMES_UNTIL_TEAM_COLOUR):
        buffer = with_analytics(system_buffer(frame))
        assert overlay.do_transform_ip(buffer) == Gst.FlowReturn.OK
    pixels = buffer.extract_dup(0, frame.nbytes)
    return np.frombuffer(pixels, np.uint8).reshape(frame.shape)


def draw_device_frames(frame):
    uploaded = uploaded_sample(frame).get_buffer()
    overlay = football_overlay(DEVICE_CAPS)
    for _ in range(FRAMES_UNTIL_TEAM_COLOUR):
        buffer = with_analytics(uploaded.copy_deep())
        flow = on_a_streaming_thread(overlay.do_transform_ip, buffer)
        assert flow == Gst.FlowReturn.OK
    return on_a_streaming_thread(device_pixels, buffer, WIDTH, HEIGHT)


@pytest.fixture(scope="module")
def drawn_frames():
    frame = pitch_frame()
    return frame, draw_device_frames(frame), draw_system_frames(frame)


def changed_pixels(drawn, original):
    return (drawn[..., :RGB_CHANNELS] != original[..., :RGB_CHANNELS]).any(axis=-1)


def test_the_device_frame_changes_where_the_system_frame_changes(drawn_frames):
    original, device_frame, system_frame = drawn_frames
    device_changed = changed_pixels(device_frame, original)
    system_changed = changed_pixels(system_frame, original)
    mismatched = np.count_nonzero(device_changed ^ system_changed)

    assert np.count_nonzero(device_changed) > 0
    assert mismatched <= DRAWN_PIXEL_MISMATCH_FRACTION * WIDTH * HEIGHT


def test_the_ellipse_under_a_red_jersey_is_red_on_the_device(drawn_frames):
    _, device_frame, _ = drawn_frames
    x, y, width, height = PLAYER_BOX
    below_the_feet = (y + height + int(ELLIPSE_HEIGHT_RATIO * width), x + width // 2)

    assert tuple(device_frame[below_the_feet][:RGB_CHANNELS]) == RED_TEAM_RGB


def test_the_hud_is_the_same_bitmap_on_both_frames(drawn_frames):
    _, device_frame, system_frame = drawn_frames
    device_hud = device_frame[HUD_REGION]

    assert (device_hud[..., :RGB_CHANNELS] == HUD_BACKGROUND_RGB).all(-1).any()
    assert np.array_equal(device_hud, system_frame[HUD_REGION])


def test_a_shared_device_buffer_is_refused():
    sample = uploaded_sample(pitch_frame())
    overlay = football_overlay(DEVICE_CAPS)

    # the sample still holds the buffer
    with pytest.raises(ValueError, match="writable"):
        on_a_streaming_thread(overlay.do_transform_ip, sample.get_buffer())
