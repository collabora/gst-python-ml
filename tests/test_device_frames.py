import importlib.util
import os
import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "plugins" / "python"))
os.environ["YOLO_AUTOINSTALL"] = "false"

gi = pytest.importorskip("gi")
gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

Gst.init(None)
torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

from backend import frameio  # noqa: E402
from test_exported_models import (  # noqa: E402
    PORTRAIT,
    VISIBLE_KEYPOINT_CONFIDENCE,
    YOLO_MODEL,
    YOLO_POSE_MODEL,
    YOLO_SEG_MODEL,
    YOLO_TOLERANCE_PIXELS,
    assert_same_boxes,
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and Gst.ElementFactory.find("cudaupload")),
    reason="needs a cuda device and the cudaupload element",
)

PULL_TIMEOUT = 10 * Gst.SECOND
# the device frame map ignores the cuda row pitch, these rows have none
FRAME_WIDTH, FRAME_HEIGHT = 1280, 960


@pytest.fixture(scope="module")
def portrait_frames():
    pipeline = Gst.parse_launch(
        f"filesrc location={PORTRAIT} ! jpegdec ! videoconvert ! videoscale "
        f"! video/x-raw,format=RGBA,width={FRAME_WIDTH},height={FRAME_HEIGHT} "
        "! tee name=split "
        "split. ! queue ! cudaupload "
        "! video/x-raw(memory:CUDAMemory),format=RGBA "
        "! appsink name=device_sink sync=false "
        "split. ! queue ! appsink name=system_sink sync=false"
    )
    pipeline.set_state(Gst.State.PLAYING)
    frames = [
        read_frame(pipeline.get_by_name(sink_name))
        for sink_name in ("device_sink", "system_sink")
    ]
    yield frames
    for buffer, _ in frames:
        frameio.release_frames(buffer)
    pipeline.set_state(Gst.State.NULL)


def read_frame(sink):
    sample = sink.emit("try-pull-sample", PULL_TIMEOUT)
    assert sample is not None
    buffer = sample.get_buffer()
    frame, _, _ = frameio.read_frames(
        buffer, sink.get_static_pad("sink"), FRAME_WIDTH, FRAME_HEIGHT
    )
    return buffer, frame


def loaded_element(element_class, model_name, track=False, tensorrt=False):
    element = element_class()
    element.set_property("model-name", model_name)
    element.set_property("engine-name", "pytorch")
    element.set_property("device", "cuda")
    element.set_property("track", track)
    element.set_property("tensorrt", tensorrt)
    element.do_load_model()
    return element


def device_and_system_results(
    element_class, model_name, frames, track=False, tensorrt=False
):
    (_, device_frame), (_, system_frame) = frames
    assert device_frame.device.type == "cuda"
    # the direct detector hands back its results on the cpu
    return (
        loaded_element(element_class, model_name, track, tensorrt)
        .do_forward(device_frame)
        .cpu(),
        loaded_element(element_class, model_name, track).do_forward(system_frame).cpu(),
    )


@pytest.mark.parametrize("track", [False, True])
def test_yolo_on_a_cuda_frame_finds_the_boxes_of_the_system_frame(
    portrait_frames, track
):
    from yolo import YOLOTransform

    device_result, system_result = device_and_system_results(
        YOLOTransform, YOLO_MODEL, portrait_frames, track
    )

    assert_same_boxes(device_result, system_result)
    assert device_result.boxes.is_track == system_result.boxes.is_track == track


def test_pose_on_a_cuda_frame_finds_the_keypoints_of_the_system_frame(
    portrait_frames,
):
    from pose import YOLOPoseTransform

    device_result, system_result = device_and_system_results(
        YOLOPoseTransform, YOLO_POSE_MODEL, portrait_frames
    )

    device_order, system_order = assert_same_boxes(device_result, system_result)
    system_keypoints = system_result.keypoints[system_order]
    visible = system_keypoints.conf > VISIBLE_KEYPOINT_CONFIDENCE
    assert visible.any()
    keypoint_difference = (
        device_result.keypoints[device_order].xy[visible] - system_keypoints.xy[visible]
    )
    assert keypoint_difference.abs().max() < YOLO_TOLERANCE_PIXELS


def test_yolo_on_a_cuda_frame_never_synchronizes_the_whole_device(
    portrait_frames, monkeypatch
):
    from yolo import YOLOTransform

    (_, device_frame), _ = portrait_frames
    element = loaded_element(YOLOTransform, YOLO_MODEL)

    def refuse_to_synchronize(*args, **kwargs):
        raise AssertionError("the direct detector synchronized the whole device")

    monkeypatch.setattr(torch.cuda, "synchronize", refuse_to_synchronize)
    assert len(element.do_forward(device_frame).boxes) > 0


@pytest.mark.skipif(
    importlib.util.find_spec("tensorrt") is None, reason="needs tensorrt"
)
@pytest.mark.parametrize(
    "module_name, class_name, model_name",
    [
        ("yolo", "YOLOTransform", YOLO_MODEL),
        ("pose", "YOLOPoseTransform", YOLO_POSE_MODEL),
    ],
)
def test_tensorrt_on_a_cuda_frame_finds_the_boxes_of_the_system_frame(
    portrait_frames, module_name, class_name, model_name
):
    element_class = getattr(importlib.import_module(module_name), class_name)

    device_result, system_result = device_and_system_results(
        element_class, model_name, portrait_frames, tensorrt=True
    )

    assert_same_boxes(device_result, system_result)


def test_yolo_segmentation_refuses_a_cuda_frame(portrait_frames):
    from yolo import YOLOTransform

    (_, device_frame), _ = portrait_frames
    element = loaded_element(YOLOTransform, YOLO_SEG_MODEL)

    with pytest.raises(ValueError, match="segment"):
        element.do_forward(device_frame)
