import sys
from concurrent.futures import ThreadPoolExecutor
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
from backend.gst.cuda_memory import map_device_frame  # noqa: E402
from backend.gst.video_transform import (  # noqa: E402
    CUDA_MEMORY_CAPS,
    CUDA_MEMORY_FEATURE,
)

needs_cuda = pytest.mark.skipif(
    not (torch.cuda.is_available() and Gst.ElementFactory.find("cudaupload")),
    reason="needs a cuda device and the cudaupload element",
)

WIDTH, HEIGHT = 640, 480
PADDED_WIDTH, PADDED_HEIGHT = 800, 600
RGB_CHANNELS = 3
RGBA_CHANNELS = 4
WRITTEN_ROW = 7
WRITTEN_VALUE = 201
FRAME_COUNT = 3
PULL_TIMEOUT = 10 * Gst.SECOND


def start_pipeline(video_format, device_format, width=WIDTH, height=HEIGHT):
    pipeline = Gst.parse_launch(
        f"videotestsrc num-buffers={FRAME_COUNT} "
        f"! video/x-raw,format={video_format},width={width},height={height} "
        "! tee name=split "
        "split. ! queue ! cudaupload ! cudaconvert "
        f"! video/x-raw(memory:CUDAMemory),format={device_format} "
        "! appsink name=device_sink sync=false "
        "split. ! queue ! appsink name=system_sink sync=false"
    )
    pipeline.set_state(Gst.State.PLAYING)
    pipeline.frame_size = (width, height)
    return pipeline


def pull_buffer(sink):
    sample = sink.emit("try-pull-sample", PULL_TIMEOUT)
    assert sample is not None
    return sample.get_buffer()


def read_frame(sink, buffer, width=WIDTH, height=HEIGHT):
    frame, num_sources, _ = frameio.read_frames(
        buffer, sink.get_static_pad("sink"), width, height
    )
    assert num_sources == 1
    return frame


@pytest.fixture
def running_pipeline(request):
    pipeline = start_pipeline(*request.param)
    yield pipeline
    pipeline.set_state(Gst.State.NULL)


@needs_cuda
@pytest.mark.parametrize(
    "running_pipeline",
    [
        ("RGBA", "RGBA"),
        ("BGRA", "BGRA"),
        # cudaconvert writes the swizzled frame on a cuda stream of its own
        ("RGBA", "BGRA"),
        # an 800 pixel row is padded to the cuda pitch
        ("RGBA", "RGBA", PADDED_WIDTH, PADDED_HEIGHT),
    ],
    indirect=True,
)
def test_a_cuda_frame_reads_without_a_copy_as_the_system_frame(running_pipeline):
    device_sink = running_pipeline.get_by_name("device_sink")
    system_sink = running_pipeline.get_by_name("system_sink")
    width, height = running_pipeline.frame_size
    for _ in range(FRAME_COUNT):
        device_buffer = pull_buffer(device_sink)
        system_buffer = pull_buffer(system_sink)

        # a streaming thread has no cuda context current
        with ThreadPoolExecutor(max_workers=1) as streaming_thread:
            device_frame = streaming_thread.submit(
                read_frame, device_sink, device_buffer, width, height
            ).result()
        system_frame = read_frame(system_sink, system_buffer, width, height)

        assert device_frame.device.type == "cuda"
        assert device_frame.shape == (height, width, RGB_CHANNELS)
        assert np.array_equal(device_frame.cpu().numpy(), system_frame)
        (device_map,) = frameio.read_map_infos[device_buffer]
        assert device_map.tensor.data_ptr() == device_map.map_info.data

        frameio.release_frames(device_buffer)
        frameio.release_frames(system_buffer)
        assert len(frameio.read_map_infos.get(device_buffer, [])) == 0


@needs_cuda
@pytest.mark.parametrize("running_pipeline", [("RGBA", "RGBA")], indirect=True)
def test_writing_a_frame_into_cuda_memory_is_refused(running_pipeline):
    device_buffer = pull_buffer(running_pipeline.get_by_name("device_sink"))

    with pytest.raises(NotImplementedError):
        frameio.write_frame(
            device_buffer, np.zeros((HEIGHT, WIDTH, RGB_CHANNELS), np.uint8)
        )


def write_then_read_back(buffer):
    writable_map = map_device_frame(buffer, HEIGHT, WIDTH, RGBA_CHANNELS, writable=True)
    writable_map.tensor[WRITTEN_ROW].fill_(WRITTEN_VALUE)
    torch.cuda.synchronize()
    writable_map.unmap()
    read_map = map_device_frame(buffer, HEIGHT, WIDTH, RGBA_CHANNELS)
    try:
        return read_map.tensor[WRITTEN_ROW].cpu()
    finally:
        read_map.unmap()


@needs_cuda
@pytest.mark.parametrize("running_pipeline", [("RGBA", "RGBA")], indirect=True)
def test_a_writable_device_map_round_trips(running_pipeline):
    pulled = pull_buffer(running_pipeline.get_by_name("device_sink"))
    device_buffer = pulled.copy_deep()

    with ThreadPoolExecutor(max_workers=1) as streaming_thread:
        row = streaming_thread.submit(write_then_read_back, device_buffer).result()

    assert (row == WRITTEN_VALUE).all()


def offers_cuda_memory(caps):
    return any(
        caps.get_features(index).contains(CUDA_MEMORY_FEATURE)
        for index in range(caps.get_size())
    )


@pytest.mark.parametrize(
    "engine_name, offers_cuda", [("onnx", False), ("pytorch", True)]
)
@pytest.mark.parametrize("direction", [Gst.PadDirection.SINK, Gst.PadDirection.SRC])
def test_only_an_engine_that_reads_device_frames_offers_cuda_caps(
    engine_name, offers_cuda, direction
):
    from classifier import Classifier

    element = Classifier()
    element.set_property("engine-name", engine_name)
    cuda_caps = Gst.Caps.from_string(CUDA_MEMORY_CAPS)

    offered = element.do_transform_caps(direction, cuda_caps, None)

    assert offers_cuda_memory(offered) == offers_cuda


@needs_cuda
def test_the_tracker_passes_cuda_memory_through():
    from tracker import TrackerTransform

    Gst.Element.register(None, "trackerundertest", Gst.Rank.NONE, TrackerTransform)
    pipeline = Gst.parse_launch(
        f"videotestsrc num-buffers={FRAME_COUNT} "
        f"! video/x-raw,format=RGBA,width={WIDTH},height={HEIGHT} ! cudaupload "
        f"! {CUDA_MEMORY_CAPS} ! trackerundertest ! {CUDA_MEMORY_CAPS} ! fakesink"
    )
    pipeline.set_state(Gst.State.PLAYING)
    message = pipeline.get_bus().timed_pop_filtered(
        PULL_TIMEOUT, Gst.MessageType.EOS | Gst.MessageType.ERROR
    )
    pipeline.set_state(Gst.State.NULL)

    assert message is not None and message.type == Gst.MessageType.EOS


@needs_cuda
@pytest.mark.parametrize("running_pipeline", [("RGBA", "RGBA")], indirect=True)
def test_a_chunk_appended_after_the_frame_keeps_the_buffer_on_the_device(
    running_pipeline,
):
    from backend.gst.cuda_memory import is_cuda_buffer

    device_sink = running_pipeline.get_by_name("device_sink")
    system_sink = running_pipeline.get_by_name("system_sink")
    device_buffer = pull_buffer(device_sink)
    system_frame = read_frame(system_sink, pull_buffer(system_sink))
    chunked = Gst.Buffer.new()
    chunked.append_memory(device_buffer.peek_memory(0))
    chunked.append_memory(Gst.Buffer.new_wrapped(b"GST-ANALYTICS:{}").peek_memory(0))

    assert is_cuda_buffer(chunked)
    device_frame = read_frame(device_sink, chunked)
    assert device_frame.device.type == "cuda"
    assert np.array_equal(device_frame.cpu().numpy(), system_frame)
    frameio.release_frames(chunked)
