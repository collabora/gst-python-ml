import sys
from pathlib import Path

import numpy as np
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "plugins" / "python"))

gi = pytest.importorskip("gi")
gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

Gst.init(None)
torch = pytest.importorskip("torch")

from backend import frameio  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and Gst.ElementFactory.find("cudaupload")),
    reason="needs a cuda device and the cudaupload element",
)

WIDTH, HEIGHT = 640, 480
RGB_CHANNELS = 3
FRAME_COUNT = 3
PULL_TIMEOUT = 10 * Gst.SECOND


def start_pipeline(video_format):
    pipeline = Gst.parse_launch(
        f"videotestsrc num-buffers={FRAME_COUNT} "
        f"! video/x-raw,format={video_format},width={WIDTH},height={HEIGHT} "
        "! tee name=split "
        "split. ! queue ! cudaupload "
        f"! video/x-raw(memory:CUDAMemory),format={video_format} "
        "! appsink name=device_sink sync=false "
        "split. ! queue ! appsink name=system_sink sync=false"
    )
    pipeline.set_state(Gst.State.PLAYING)
    return pipeline


def pull_buffer(sink):
    sample = sink.emit("try-pull-sample", PULL_TIMEOUT)
    assert sample is not None
    return sample.get_buffer()


def read_frame(sink, buffer):
    frame, num_sources, _ = frameio.read_frames(
        buffer, sink.get_static_pad("sink"), WIDTH, HEIGHT
    )
    assert num_sources == 1
    return frame


@pytest.fixture
def running_pipeline(request):
    pipeline = start_pipeline(request.param)
    yield pipeline
    pipeline.set_state(Gst.State.NULL)


@pytest.mark.parametrize("running_pipeline", ["RGBA", "BGRA"], indirect=True)
def test_a_cuda_frame_reads_without_a_copy_as_the_system_frame(running_pipeline):
    device_sink = running_pipeline.get_by_name("device_sink")
    system_sink = running_pipeline.get_by_name("system_sink")
    for _ in range(FRAME_COUNT):
        device_buffer = pull_buffer(device_sink)
        system_buffer = pull_buffer(system_sink)

        device_frame = read_frame(device_sink, device_buffer)
        system_frame = read_frame(system_sink, system_buffer)

        assert device_frame.device.type == "cuda"
        assert device_frame.shape == (HEIGHT, WIDTH, RGB_CHANNELS)
        assert np.array_equal(device_frame.cpu().numpy(), system_frame)
        (device_map,) = frameio.read_map_infos[device_buffer]
        assert device_map.tensor.data_ptr() == device_map.map_info.data

        frameio.release_frames(device_buffer)
        frameio.release_frames(system_buffer)
        assert len(frameio.read_map_infos.get(device_buffer, [])) == 0


@pytest.mark.parametrize("running_pipeline", ["RGBA"], indirect=True)
def test_writing_a_frame_into_cuda_memory_is_refused(running_pipeline):
    device_buffer = pull_buffer(running_pipeline.get_by_name("device_sink"))

    with pytest.raises(NotImplementedError):
        frameio.write_frame(
            device_buffer, np.zeros((HEIGHT, WIDTH, RGB_CHANNELS), np.uint8)
        )
