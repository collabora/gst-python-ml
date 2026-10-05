import os
import sys
from pathlib import Path

import numpy as np
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
PLUGIN_DIR = BASE_DIR / "plugins" / "python"
# the plugin loader reads GST_PLUGIN_PATH at Gst.init
os.environ["GST_PLUGIN_PATH"] = str(BASE_DIR / "plugins")
sys.path.insert(0, str(PLUGIN_DIR))

gi = pytest.importorskip("gi")
gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

Gst.init(None)

PORTRAIT = BASE_DIR / "data" / "Chinedu-Obasi_2684938.jpg"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
# the exported graph takes a square frame
DEPTH_CORRELATION_FLOOR = 0.98
CLIP_MODELS = ["openai/clip-vit-base-patch32", "google/siglip-base-patch16-224"]
CLIP_LABELS = ["a man", "a dog", "a car", "a tree"]
CLIP_PROBABILITY_TOLERANCE = 0.02
ANOMALY_BACKBONE = "resnet18"
HEATMAP_TOLERANCE = 0.01
EMBEDDING_MODELS = ["openai/clip-vit-base-patch32", "facebook/dinov2-small"]
EMBEDDING_COSINE_FLOOR = 0.999
ACTION_MODEL = "MCG-NJU/videomae-base-finetuned-kinetics"
ACTION_WINDOW_FRAMES = 16
ACTION_SCORE_TOLERANCE = 0.02
SUPERRES_MODEL = "real-esrgan-x2"
# the 2x model needs odd sides padded
SUPERRES_FRAME_SIZE = (161, 121)
SUPERRES_MEAN_DIFFERENCE = 1.0
FLOW_MODEL = "raft_small"
# the size the exported graph takes
FLOW_FRAME_SIZE = (640, 360)
FLOW_SHIFT_PIXELS = 8
FLOW_TOLERANCE_PIXELS = 0.5
# the task engine first
ENGINE_NAMES = ("pytorch", "onnx")

pytest.importorskip("onnxruntime")
pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def portrait_rgb():
    import cv2

    return cv2.cvtColor(cv2.imread(str(PORTRAIT)), cv2.COLOR_BGR2RGB)


def loaded_element(element_class, model_name, engine_name):
    element = element_class()
    element.set_property("model-name", model_name)
    element.set_property("engine-name", engine_name)
    element.do_load_model()
    return element


def test_depth_on_onnx_matches_depth_on_pytorch(portrait_rgb):
    from depth import DepthTransform

    reference, exported = (
        loaded_element(DepthTransform, DEPTH_MODEL, engine_name).forward(portrait_rgb)
        for engine_name in ENGINE_NAMES
    )

    assert exported.shape == reference.shape == portrait_rgb.shape[:2]
    correlation = np.corrcoef(exported.ravel(), reference.ravel())[0, 1]
    assert correlation > DEPTH_CORRELATION_FLOOR


def clip_probabilities(model_name, engine_name, frame):
    from clip import CLIPTransform

    element = loaded_element(CLIPTransform, model_name, engine_name)
    element.task_engine.clip_labels = CLIP_LABELS
    return dict(element.task_engine.do_forward(frame))


@pytest.mark.parametrize("model_name", CLIP_MODELS)
def test_clip_on_onnx_matches_clip_on_pytorch(model_name, portrait_rgb):
    reference, exported = (
        clip_probabilities(model_name, engine_name, portrait_rgb)
        for engine_name in ENGINE_NAMES
    )

    assert max(reference, key=reference.get) == max(exported, key=exported.get)
    for label, probability in reference.items():
        assert exported[label] == pytest.approx(
            probability, abs=CLIP_PROBABILITY_TOLERANCE
        )


def test_anomaly_on_onnx_matches_anomaly_on_pytorch(portrait_rgb):
    from anomaly import AnomalyTransform

    reference, exported = (
        loaded_element(AnomalyTransform, ANOMALY_BACKBONE, engine_name).forward(
            portrait_rgb
        )
        for engine_name in ENGINE_NAMES
    )

    assert exported["heatmap"] == pytest.approx(
        reference["heatmap"], abs=HEATMAP_TOLERANCE
    )


@pytest.mark.parametrize("model_name", EMBEDDING_MODELS)
def test_embedding_on_onnx_matches_embedding_on_pytorch(model_name, portrait_rgb):
    from embedding import EmbeddingTransform

    reference, exported = (
        loaded_element(EmbeddingTransform, model_name, engine_name).forward(
            portrait_rgb
        )
        for engine_name in ENGINE_NAMES
    )

    assert exported.shape == reference.shape
    assert float(exported @ reference) > EMBEDDING_COSINE_FLOOR


def test_action_on_onnx_matches_action_on_pytorch(portrait_rgb):
    from action import ActionTransform

    window = [portrait_rgb] * ACTION_WINDOW_FRAMES
    reference, exported = (
        loaded_element(ActionTransform, ACTION_MODEL, engine_name).forward(window)
        for engine_name in ENGINE_NAMES
    )

    assert exported["label"] == reference["label"]
    assert exported["score"] == pytest.approx(
        reference["score"], abs=ACTION_SCORE_TOLERANCE
    )


def test_superres_on_onnx_matches_superres_on_pytorch(portrait_rgb):
    import cv2
    from superres import SuperResTransform

    small_frame = cv2.resize(portrait_rgb, SUPERRES_FRAME_SIZE)
    reference, exported = (
        loaded_element(SuperResTransform, SUPERRES_MODEL, engine_name).forward(
            small_frame
        )
        for engine_name in ENGINE_NAMES
    )

    assert exported.shape == reference.shape
    difference = np.abs(exported.astype(int) - reference.astype(int))
    assert difference.mean() < SUPERRES_MEAN_DIFFERENCE


def test_optical_flow_on_onnx_matches_optical_flow_on_pytorch(portrait_rgb):
    import cv2
    from optical_flow import OpticalFlowTransform

    previous_frame = cv2.resize(portrait_rgb, FLOW_FRAME_SIZE)
    current_frame = np.roll(previous_frame, FLOW_SHIFT_PIXELS, axis=1)
    reference, exported = (
        loaded_element(OpticalFlowTransform, FLOW_MODEL, engine_name).forward(
            previous_frame, current_frame
        )
        for engine_name in ENGINE_NAMES
    )

    assert np.median(reference[..., 0]) == pytest.approx(
        FLOW_SHIFT_PIXELS, abs=FLOW_TOLERANCE_PIXELS
    )
    assert np.abs(exported - reference).mean() < FLOW_TOLERANCE_PIXELS
