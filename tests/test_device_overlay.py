import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "plugins" / "python"))

gi = pytest.importorskip("gi")
gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import Gst, GstVideo  # noqa: E402

Gst.init(None)
torch = pytest.importorskip("torch")
pytest.importorskip("cairo")

from backend import analytics, frameio  # noqa: E402
from backend.gst.cuda_memory import map_device_frame  # noqa: E402

needs_cuda = pytest.mark.skipif(
    not (torch.cuda.is_available() and Gst.ElementFactory.find("cudaupload")),
    reason="needs a cuda device and the cudaupload element",
)

WIDTH, HEIGHT = 640, 480
SYSTEM_CAPS = f"video/x-raw,format=RGBA,width={WIDTH},height={HEIGHT}"
DEVICE_CAPS = (
    f"video/x-raw(memory:CUDAMemory),format=RGBA,width={WIDTH},height={HEIGHT}"
)
PULL_TIMEOUT = 10 * Gst.SECOND
RGBA_CHANNELS = 4
RGB_CHANNELS = 3
FULL_INTENSITY = 255
BOX_LINE_WIDTH = 2
LABEL_OFFSET = 10
LABEL_HEIGHT = 12
# (label, x, y, width, height)
TRACKED_OBJECTS = (
    ("stream_0_person_id_1", 100, 120, 160, 200),
    ("stream_0_car_id_2", 360, 200, 180, 140),
)
TRAIL_POINTS = ((40, 420), (300, 420))
TRAIL_LABEL = "stream_0_ball_trail"
RED = (FULL_INTENSITY, 0, 0)
YELLOW = (FULL_INTENSITY, FULL_INTENSITY, 0)
BLACK = (0, 0, 0)
# cairo antialiases edges and text and uses another font
DRAWN_PIXEL_MISMATCH_FRACTION = 0.003


def pull_tee_samples():
    pipeline = Gst.parse_launch(
        f"videotestsrc pattern=black num-buffers=1 ! {SYSTEM_CAPS} ! tee name=split "
        f"split. ! queue ! cudaupload ! cudaconvert ! {DEVICE_CAPS} "
        "! appsink name=device_sink sync=false "
        "split. ! queue ! appsink name=system_sink sync=false"
    )
    pipeline.set_state(Gst.State.PLAYING)
    samples = {}
    try:
        for name in ("device_sink", "system_sink"):
            samples[name] = pipeline.get_by_name(name).emit(
                "try-pull-sample", PULL_TIMEOUT
            )
            assert samples[name] is not None
    finally:
        pipeline.set_state(Gst.State.NULL)
    return samples


def with_analytics(buffer):
    writable = buffer.copy_deep()
    meta = analytics.add_relation_meta(writable)
    for label, x, y, width, height in TRACKED_OBJECTS:
        analytics.add_object(meta, label, x, y, width, height, 1.0)
    for x, y in TRAIL_POINTS:
        analytics.add_object(meta, TRAIL_LABEL, x, y, 0, 0, 1.0)
    return writable


def draw(buffer, caps):
    from overlay import Overlay

    overlay = Overlay()
    parsed_caps = Gst.Caps.from_string(caps)
    overlay.do_set_caps(parsed_caps, parsed_caps)
    return overlay.do_transform_ip(buffer)


def device_pixels(buffer):
    device_map = map_device_frame(buffer, HEIGHT, WIDTH, RGBA_CHANNELS)
    try:
        return device_map.tensor[..., :RGB_CHANNELS].cpu().numpy()
    finally:
        device_map.unmap()


def system_pixels(buffer):
    frame_size = WIDTH * HEIGHT * RGBA_CHANNELS
    frame = np.frombuffer(buffer.extract_dup(0, frame_size), np.uint8)
    return frame.reshape(HEIGHT, WIDTH, RGBA_CHANNELS)[..., :RGB_CHANNELS]


def on_a_streaming_thread(function, *arguments):
    # a streaming thread has no cuda context current
    with ThreadPoolExecutor(max_workers=1) as streaming_thread:
        return streaming_thread.submit(function, *arguments).result()


@pytest.fixture(scope="module")
def drawn_frames():
    samples = pull_tee_samples()
    device_buffer = with_analytics(samples["device_sink"].get_buffer())
    system_buffer = with_analytics(samples["system_sink"].get_buffer())
    # the cairo path draws only on a buffer with a video meta
    GstVideo.buffer_add_video_meta(
        system_buffer,
        GstVideo.VideoFrameFlags.NONE,
        GstVideo.VideoFormat.RGBA,
        WIDTH,
        HEIGHT,
    )

    assert on_a_streaming_thread(draw, device_buffer, DEVICE_CAPS) == Gst.FlowReturn.OK
    assert draw(system_buffer, SYSTEM_CAPS) == Gst.FlowReturn.OK

    device_frame = on_a_streaming_thread(device_pixels, device_buffer)
    return device_frame, system_pixels(system_buffer)


@needs_cuda
def test_box_edges_are_drawn_in_the_box_colour(drawn_frames):
    device_frame, _ = drawn_frames
    for _, x, y, width, height in TRACKED_OBJECTS:
        right, bottom = x + width, y + height
        outer_left, outer_top = x - BOX_LINE_WIDTH // 2, y - BOX_LINE_WIDTH // 2
        edge_rows = range(outer_top, outer_top + BOX_LINE_WIDTH)
        bottom_rows = range(bottom - 1, bottom - 1 + BOX_LINE_WIDTH)
        edge_columns = range(outer_left, outer_left + BOX_LINE_WIDTH)
        right_columns = range(right - 1, right - 1 + BOX_LINE_WIDTH)
        for row in (*edge_rows, *bottom_rows):
            assert (device_frame[row, outer_left : right + 1] == RED).all()
        for column in (*edge_columns, *right_columns):
            assert (device_frame[outer_top : bottom + 1, column] == RED).all()
        assert tuple(device_frame[outer_top - 1, x + width // 2]) == BLACK
        assert tuple(device_frame[y + 1, x + width // 2]) == BLACK
        assert tuple(device_frame[bottom + 1, x + width // 2]) == BLACK


@needs_cuda
def test_the_label_is_drawn_in_red_above_the_box(drawn_frames):
    device_frame, _ = drawn_frames
    for _, x, y, width, _ in TRACKED_OBJECTS:
        baseline = y - LABEL_OFFSET
        label = device_frame[baseline - LABEL_HEIGHT : baseline, x : x + width]
        assert (label[..., 1:] == 0).all()
        assert label[..., 0].max() == FULL_INTENSITY


@needs_cuda
def test_the_trail_is_drawn_in_yellow(drawn_frames):
    device_frame, _ = drawn_frames
    (start_x, y), (end_x, _) = TRAIL_POINTS
    assert (device_frame[y, start_x:end_x] == YELLOW).all()


@needs_cuda
def test_the_device_frame_draws_where_cairo_draws(drawn_frames):
    device_frame, system_frame = drawn_frames
    device_drawn = (device_frame != 0).any(axis=-1)
    cairo_drawn = (system_frame != 0).any(axis=-1)
    mismatched = np.count_nonzero(device_drawn ^ cairo_drawn)

    assert np.count_nonzero(device_drawn) > 0
    assert mismatched <= DRAWN_PIXEL_MISMATCH_FRACTION * WIDTH * HEIGHT


@needs_cuda
def test_drawing_on_a_shared_buffer_is_refused():
    sample = pull_tee_samples()["device_sink"]

    # the sample still holds the buffer
    with pytest.raises(ValueError, match="writable"):
        frameio.read_frame_for_drawing(sample.get_buffer(), WIDTH, HEIGHT, "RGBA")
