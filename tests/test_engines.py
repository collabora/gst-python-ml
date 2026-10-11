import os
import shutil
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

DETECTOR_WEIGHTS = "yolo11n.pt"
DETECTOR_INPUT_SIZE = 640
DETECTION_FRAME_COUNT = 10
DETECTION_FRAME_STRIDE = 10
CONFIDENCE_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.45
MATCH_IOU_THRESHOLD = 0.5
MINIMUM_MATCHED_FRACTION = 0.8

GGUF_REPOSITORY = "ggml-org/models"
GGUF_FILE = "tinyllamas/stories260K.gguf"
GGUF_QUANT = "stories260K"
CAUSAL_LM = "HuggingFaceTB/SmolLM2-135M-Instruct"
GENERATION_PROMPT = "Once upon a time"
GENERATION_TOKENS = 8

WHISPER_MODEL = "openai/whisper-tiny"
WHISPER_CLIP = BASE_DIR / "data" / "air_traffic_korean_with_english.wav"
WHISPER_CLIP_SECONDS = 20
WHISPER_LANGUAGE = "ko"
# the clip opens with 항공 관제, air traffic control
WHISPER_EXPECTED_WORD = "항공"
PCM16_SCALE = 32768.0

# the tensorflow exports take channels last, the rest take channels first like the PIPELINES.md pipelines
DETECTION_ENGINES = [
    ("onnx", "onnxruntime", "onnx", ".onnx", "nchw"),
    ("openvino", "openvino", "openvino", ".xml", "nchw"),
    ("tensorflow", "tensorflow", "saved_model", None, "auto"),
    # the ultralytics tflite export installs litert-torch, which pins torch
    ("tflite", "ai_edge_litert", "onnx", ".onnx", "auto"),
    ("ncnn", "ncnn", "ncnn", ".param", "nchw"),
    ("executorch", "executorch", "executorch", ".pte", "nchw"),
    ("iree", "iree.runtime", "onnx", ".onnx", "nchw"),
    ("tinygrad", "tinygrad", "onnx", ".onnx", "auto"),
    ("migraphx", "migraphx", "onnx", ".onnx", "nchw"),
]

# what an engine needs to convert the export beyond its own package
CONVERSION_MODULES = {"tflite": "tensorflow"}

DRPAI_EMULATION_DIR = BASE_DIR / "extern" / "rzv2h" / "emulation"


def require_module(engine_name, module_name):
    try:
        __import__(module_name)
    except ImportError:
        if REQUIRED_ENGINE == engine_name:
            pytest.fail(
                f"{module_name} is not importable but {engine_name} is required"
            )
        pytest.skip(f"{module_name} not installed")


def engine_on_cpu(name, module_name):
    require_module(name, module_name)
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
def exported_detector(detector, tmp_path_factory):
    exported = {}

    def export(export_format):
        if export_format not in exported:
            # ultralytics drops onnx2tf's calibration file in the working directory
            previous_directory = os.getcwd()
            os.chdir(tmp_path_factory.mktemp("export"))
            try:
                exported[export_format] = Path(
                    detector.export(format=export_format, imgsz=DETECTOR_INPUT_SIZE)
                )
            finally:
                os.chdir(previous_directory)
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
    if engine_name in CONVERSION_MODULES:
        require_module(engine_name, CONVERSION_MODULES[engine_name])
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


def test_drpai_detections_match_reference_on_the_emulated_runtime(
    monkeypatch, tmp_path, exported_detector, reference_boxes, people_frames_rgb
):
    # the stand-in runs the onnx file in the model directory with onnxruntime
    monkeypatch.syspath_prepend(str(DRPAI_EMULATION_DIR))
    engine = engine_on_cpu("drpai", "onnxruntime")
    shutil.copy(exported_detector("onnx"), tmp_path)
    assert engine.do_load_model(str(tmp_path), imgsz=DETECTOR_INPUT_SIZE) is True

    candidate_boxes = {
        index: detections_as_boxes(engine.do_forward(frame))
        for index, frame in enumerate(people_frames_rgb)
    }
    fraction = matched_box_fraction(
        reference_boxes, candidate_boxes, MATCH_IOU_THRESHOLD
    )
    assert fraction is not None, "the reference run found no people"
    assert fraction >= MINIMUM_MATCHED_FRACTION, f"matched {fraction:.2f}"


def test_llamacpp_generates_text():
    from huggingface_hub import hf_hub_download

    engine = engine_on_cpu("llamacpp", "llama_cpp")
    model_path = hf_hub_download(GGUF_REPOSITORY, GGUF_FILE)
    assert engine.do_load_model(model_path) is True
    text = engine.do_generate(GENERATION_PROMPT, max_length=GENERATION_TOKENS)
    assert isinstance(text, str) and text.strip()


def test_llamacpp_downloads_a_hub_gguf_by_repo_and_quant():
    engine = engine_on_cpu("llamacpp", "llama_cpp")
    assert engine.do_load_model(f"{GGUF_REPOSITORY}:{GGUF_QUANT}") is True
    text = engine.do_generate(GENERATION_PROMPT, max_length=GENERATION_TOKENS)
    assert isinstance(text, str) and text.strip()


def test_onnx_generates_text_from_a_hub_causal_lm():
    engine = engine_on_cpu("onnx", "onnxruntime_genai")
    assert engine.do_load_model(CAUSAL_LM) is True
    text = engine.do_generate(GENERATION_PROMPT, max_length=GENERATION_TOKENS)
    assert isinstance(text, str) and text.strip()


def test_openvino_generates_text_from_a_hub_causal_lm():
    engine = engine_on_cpu("openvino", "openvino_genai")
    assert engine.do_load_model(CAUSAL_LM) is True
    text = engine.do_generate(GENERATION_PROMPT, max_length=GENERATION_TOKENS)
    assert isinstance(text, str) and text.strip()


def whisper_clip():
    import wave

    with wave.open(str(WHISPER_CLIP)) as reader:
        rate = reader.getframerate()
        frames = reader.readframes(rate * WHISPER_CLIP_SECONDS)
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / PCM16_SCALE


def transcribed_with(engine):
    assert engine.do_load_model(WHISPER_MODEL) is True
    texts = engine.model.transcribe(
        whisper_clip(), WHISPER_LANGUAGE, "transcribe", 1, ""
    )
    return " ".join(texts)


def test_onnx_transcribes_with_a_hub_whisper():
    engine = engine_on_cpu("onnx", "onnxruntime_genai")
    assert WHISPER_EXPECTED_WORD in transcribed_with(engine)


def test_openvino_transcribes_with_a_hub_whisper():
    engine = engine_on_cpu("openvino", "openvino_genai")
    assert WHISPER_EXPECTED_WORD in transcribed_with(engine)


def test_llamacpp_transcribes_with_a_hub_whisper():
    engine = engine_on_cpu("llamacpp", "pywhispercpp")
    assert WHISPER_EXPECTED_WORD in transcribed_with(engine)


def test_mlx_transcribes_with_a_hub_whisper():
    engine = engine_on_cpu("mlx", "mlx_whisper")
    assert WHISPER_EXPECTED_WORD in transcribed_with(engine)


def test_onnx_refuses_a_model_path_that_does_not_exist():
    engine = engine_on_cpu("onnx", "onnxruntime")
    with pytest.raises(FileNotFoundError, match="yolo11m.onxx"):
        engine.do_load_model("yolo11m.onxx")


def test_a_builtin_engine_whose_package_is_missing_raises_the_import_error(monkeypatch):
    monkeypatch.setitem(
        EngineFactory.BUILTIN_ENGINES,
        "engine_without_its_package",
        ("module_that_is_not_installed", "MissingEngine"),
    )
    with pytest.raises(ImportError, match="module_that_is_not_installed"):
        EngineFactory.create("engine_without_its_package")


def test_an_unknown_engine_name_is_refused():
    with pytest.raises(ValueError, match="Unsupported engine type: no_such_engine"):
        EngineFactory.create("no_such_engine")


def test_anomaly_transform_without_a_backbone():
    from engine.anomaly_engine import AnomalyEngine

    engine = AnomalyEngine()
    with pytest.raises(ValueError, match="anomaly backbone is not loaded"):
        engine._get_transform()
