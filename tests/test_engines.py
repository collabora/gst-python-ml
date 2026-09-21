import os
import sys
from pathlib import Path

import numpy as np
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "plugins" / "python"))

from engine.engine_factory import EngineFactory  # noqa: E402
from utils.box_matching import matched_box_fraction  # noqa: E402

# a ci matrix leg fails on a missing import instead of skipping
REQUIRED_ENGINE = os.environ.get("PYML_REQUIRE_ENGINE")

VIDEO = BASE_DIR / "data" / "people.mp4"
PORTRAIT = BASE_DIR / "data" / "Chinedu-Obasi_2684938.jpg"

DETECTOR_WEIGHTS = "yolo11n.pt"
DETECTOR_INPUT_SIZE = 640
DETECTION_FRAME_COUNT = 10
DETECTION_FRAME_STRIDE = 10
CONFIDENCE_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.45
MATCH_IOU_THRESHOLD = 0.5
MINIMUM_MATCHED_FRACTION = 0.8

CLASSIFIER_MODEL = "resnet18"
CLASSIFIER_INPUT_SIZE = 224

GGUF_REPOSITORY = "ggml-org/models"
GGUF_FILE = "tinyllamas/stories260K.gguf"
GENERATION_PROMPT = "Once upon a time"
GENERATION_TOKENS = 8

# the tensorflow exports take channels last, the rest take channels first like the README pipelines
DETECTION_ENGINES = [
    ("onnx", "onnxruntime", "onnx", ".onnx", "nchw"),
    ("openvino", "openvino", "openvino", ".xml", "nchw"),
    ("tensorflow", "tensorflow", "saved_model", None, "auto"),
    ("tflite", "tensorflow", "tflite", "_float32.tflite", "auto"),
    ("ncnn", "ncnn", "ncnn", ".param", "nchw"),
    ("executorch", "executorch", "executorch", ".pte", "nchw"),
    ("iree", "iree.runtime", "onnx", ".onnx", "nchw"),
]

CLASSIFICATION_ENGINES = [
    pytest.param("pytorch", "torch", id="pytorch"),
    pytest.param("tvm", "tvm", id="tvm"),
    pytest.param("mlx", "mlx.core", id="mlx"),
    pytest.param("jax", "jax", id="jax"),
    pytest.param("tinygrad", "tinygrad", id="tinygrad"),
]


def engine_on_cpu(name, module_name):
    try:
        __import__(module_name)
    except ImportError:
        if REQUIRED_ENGINE == name:
            pytest.fail(f"{module_name} is not importable but {name} is required")
        pytest.skip(f"{module_name} not installed")
    engine = EngineFactory.create(name)
    engine.do_set_device("cpu")
    return engine


def as_box(x1, y1, x2, y2):
    return {"x": float(x1), "y": float(y1), "w": float(x2 - x1), "h": float(y2 - y1)}


@pytest.fixture(scope="session")
def people_frames_bgr():
    import cv2

    capture = cv2.VideoCapture(str(VIDEO))
    frames = []
    index = 0
    while len(frames) < DETECTION_FRAME_COUNT:
        ok, frame = capture.read()
        if not ok:
            break
        if index % DETECTION_FRAME_STRIDE == 0:
            frames.append(cv2.resize(frame, (DETECTOR_INPUT_SIZE, DETECTOR_INPUT_SIZE)))
        index += 1
    capture.release()
    assert len(frames) == DETECTION_FRAME_COUNT
    return frames


@pytest.fixture(scope="session")
def people_frames_rgb(people_frames_bgr):
    import cv2

    return [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in people_frames_bgr]


@pytest.fixture(scope="session")
def detector(tmp_path_factory):
    from ultralytics import YOLO

    return YOLO(str(tmp_path_factory.mktemp("yolo") / DETECTOR_WEIGHTS))


@pytest.fixture(scope="session")
def reference_boxes(detector, people_frames_bgr):
    results = detector.predict(
        people_frames_bgr,
        imgsz=DETECTOR_INPUT_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        iou=NMS_IOU_THRESHOLD,
        verbose=False,
    )
    return {
        index: [as_box(*xyxy) for xyxy in result.boxes.xyxy.cpu().numpy()]
        for index, result in enumerate(results)
    }


@pytest.fixture(scope="session")
def exported_detector(detector):
    exported = {}

    def export(export_format):
        if export_format not in exported:
            exported[export_format] = Path(
                detector.export(format=export_format, imgsz=DETECTOR_INPUT_SIZE)
            )
        return exported[export_format]

    return export


def artifact_path(exported, suffix):
    if suffix is None or exported.is_file():
        return exported
    matches = sorted(exported.rglob(f"*{suffix}"))
    assert matches, f"no *{suffix} under {exported}"
    return matches[0]


def detections_as_boxes(result):
    if isinstance(result, list):
        result = result[0]
    assert isinstance(result, dict), f"detector returned {type(result).__name__}"
    return [as_box(*xyxy) for xyxy in np.asarray(result["boxes"])]


@pytest.mark.parametrize(
    "engine_name, module_name, export_format, suffix, input_format",
    DETECTION_ENGINES,
    ids=[row[0] for row in DETECTION_ENGINES],
)
def test_detections_match_reference(
    engine_name,
    module_name,
    export_format,
    suffix,
    input_format,
    exported_detector,
    reference_boxes,
    people_frames_rgb,
):
    engine = engine_on_cpu(engine_name, module_name)
    model_path = artifact_path(exported_detector(export_format), suffix)
    engine.input_format = input_format
    engine.post_process = "anchor_free"
    assert engine.do_load_model(str(model_path)) is True

    candidate_boxes = {
        index: detections_as_boxes(engine.do_forward(frame))
        for index, frame in enumerate(people_frames_rgb)
    }
    fraction = matched_box_fraction(
        reference_boxes, candidate_boxes, MATCH_IOU_THRESHOLD
    )
    assert fraction is not None, "the reference run found no people"
    assert fraction >= MINIMUM_MATCHED_FRACTION, f"matched {fraction:.2f}"


@pytest.fixture(scope="session")
def portrait_rgb():
    import cv2

    image = cv2.imread(str(PORTRAIT))
    image = cv2.resize(image, (CLASSIFIER_INPUT_SIZE, CLASSIFIER_INPUT_SIZE))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@pytest.fixture(scope="session")
def reference_class(portrait_rgb):
    import torch
    from torchvision import models

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).eval()
    image = torch.from_numpy(portrait_rgb).float().permute(2, 0, 1)[None] / 255.0
    with torch.inference_mode():
        return int(model(image).argmax())


@pytest.mark.parametrize("engine_name, module_name", CLASSIFICATION_ENGINES)
def test_top_class_matches_reference(
    engine_name, module_name, portrait_rgb, reference_class
):
    engine = engine_on_cpu(engine_name, module_name)
    assert engine.do_load_model(CLASSIFIER_MODEL) is True
    result = engine.do_forward(portrait_rgb)
    assert isinstance(result, dict), f"classifier returned {type(result).__name__}"
    assert result["labels"][0] == reference_class


def test_llamacpp_generates_text():
    from huggingface_hub import hf_hub_download

    engine = engine_on_cpu("llamacpp", "llama_cpp")
    model_path = hf_hub_download(GGUF_REPOSITORY, GGUF_FILE)
    assert engine.do_load_model(model_path) is True
    text = engine.do_generate(GENERATION_PROMPT, max_length=GENERATION_TOKENS)
    assert isinstance(text, str) and text.strip()
