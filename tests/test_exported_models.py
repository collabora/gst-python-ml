import importlib.util
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
SIGLIP_ON_TENSORFLOW = ("tensorflow", CLIP_MODELS[1])
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
SAM_MODEL = "facebook/sam2-hiera-tiny"
SAM_SCORE_TOLERANCE = 1e-3
# a handful of pixels on mask edges flip
SAM_MASK_AGREEMENT_FLOOR = 0.9999
YOLO_MODEL = "yolo11n"
YOLO_POSE_MODEL = "yolo11n-pose"
YOLO_CONFIDENCE = 0.25
# pytorch letterboxes the portrait to 640x480, the exported graph to 640x640
YOLO_TOLERANCE_PIXELS = 12.0
# an off-frame keypoint has a guessed position
VISIBLE_KEYPOINT_CONFIDENCE = 0.5
OWL_MODEL = "google/owlv2-base-patch16-ensemble"
# its image encoder takes the text
GROUNDING_MODEL = "IDEA-Research/grounding-dino-tiny"
ZERO_SHOT_LABELS = "a face, a blue chair, a dog"
ZERO_SHOT_SCORE_TOLERANCE = 0.001
ZERO_SHOT_BOX_TOLERANCE_PIXELS = 1
# the task engine first
ENGINE_NAMES = ("pytorch", "onnx")
# every engine that runs the exported onnx, in its own format or as is
EXPORTED_MODEL_ENGINE_PACKAGES = {
    "onnx": "onnxruntime",
    "openvino": "openvino",
    "tvm": "tvm",
    "tensorflow": "tensorflow",
    "tflite": "ai_edge_litert",
    "ncnn": "ncnn",
    "executorch": "executorch",
    "iree": "iree.runtime",
    "tinygrad": "tinygrad",
    "migraphx": "migraphx",
}
# jax runs a keras-hub preset instead of the export
BUILTIN_ENGINE_PACKAGES = {**EXPORTED_MODEL_ENGINE_PACKAGES, "jax": "keras_hub"}
# a ci matrix leg fails on a missing engine instead of skipping
REQUIRED_ENGINE = os.environ.get("PYML_REQUIRE_ENGINE")
# the pytorch reference stays on the cpu
BUILTIN_ENGINE_DEVICE = os.environ.get("PYML_TEST_DEVICE", "cpu")

pytest.importorskip("onnxruntime")
pytest.importorskip("transformers")


# importing keras-hub here would fix keras on its default backend
def require_engine(engine_name):
    package = BUILTIN_ENGINE_PACKAGES[engine_name]
    if importlib.util.find_spec(package) is not None:
        return
    if REQUIRED_ENGINE == engine_name:
        pytest.fail(f"{package} is not installed but {engine_name} is required")
    pytest.skip(f"{package} is not installed")


@pytest.fixture(scope="module")
def portrait_rgb():
    import cv2

    return cv2.cvtColor(cv2.imread(str(PORTRAIT)), cv2.COLOR_BGR2RGB)


def loaded_element(element_class, model_name, engine_name):
    element = element_class()
    element.set_property("model-name", model_name)
    element.set_property("engine-name", engine_name)
    if engine_name != ENGINE_NAMES[0]:
        element.set_property("device", BUILTIN_ENGINE_DEVICE)
    element.do_load_model()
    return element


@pytest.mark.parametrize("builtin_engine", BUILTIN_ENGINE_PACKAGES)
def test_depth_on_a_builtin_engine_matches_depth_on_pytorch(
    builtin_engine, portrait_rgb
):
    require_engine(builtin_engine)
    from depth import DepthTransform

    reference, exported = (
        loaded_element(DepthTransform, DEPTH_MODEL, engine_name).forward(portrait_rgb)
        for engine_name in (ENGINE_NAMES[0], builtin_engine)
    )

    assert exported.shape == reference.shape == portrait_rgb.shape[:2]
    correlation = np.corrcoef(exported.ravel(), reference.ravel())[0, 1]
    assert correlation > DEPTH_CORRELATION_FLOOR


def clip_probabilities(model_name, engine_name, frame):
    from clip import CLIPTransform

    element = loaded_element(CLIPTransform, model_name, engine_name)
    element.task_engine.clip_labels = CLIP_LABELS
    return dict(element.task_engine.do_forward(frame))


@pytest.mark.parametrize("builtin_engine", BUILTIN_ENGINE_PACKAGES)
@pytest.mark.parametrize("model_name", CLIP_MODELS)
def test_clip_on_a_builtin_engine_matches_clip_on_pytorch(
    model_name, builtin_engine, portrait_rgb
):
    require_engine(builtin_engine)
    if (builtin_engine, model_name) == SIGLIP_ON_TENSORFLOW:
        pytest.xfail("onnx2tf's tf_converter splits a constant vector one off")
    reference, exported = (
        clip_probabilities(model_name, engine_name, portrait_rgb)
        for engine_name in (ENGINE_NAMES[0], builtin_engine)
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
    pytest.importorskip("spandrel")
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


def test_sam_on_onnx_matches_sam_on_pytorch(portrait_rgb):
    from sam import SamTransform

    reference, exported = (
        loaded_element(SamTransform, SAM_MODEL, engine_name).forward(portrait_rgb)
        for engine_name in ENGINE_NAMES
    )

    assert len(exported["masks"]) == len(reference["masks"])
    for exported_mask, reference_mask in zip(exported["masks"], reference["masks"]):
        assert exported_mask["score"] == pytest.approx(
            reference_mask["score"], abs=SAM_SCORE_TOLERANCE
        )
    assert exported["raw_masks"].shape == reference["raw_masks"].shape
    agreement = (exported["raw_masks"] == reference["raw_masks"]).mean()
    assert agreement > SAM_MASK_AGREEMENT_FLOOR


def detection_results(frame, builtin_engine):
    pytest.importorskip("ultralytics")
    from yolo import YOLOTransform

    for engine_name in (ENGINE_NAMES[0], builtin_engine):
        element = loaded_element(YOLOTransform, YOLO_MODEL, engine_name)
        element.set_property("confidence", YOLO_CONFIDENCE)
        yield element.do_forward(frame)


def pose_results(frame, builtin_engine):
    pytest.importorskip("ultralytics")
    from pose import YOLOPoseTransform

    for engine_name in (ENGINE_NAMES[0], builtin_engine):
        element = loaded_element(YOLOPoseTransform, YOLO_POSE_MODEL, engine_name)
        yield element.do_forward(frame)


def assert_same_boxes(exported, reference):
    import torch

    assert len(reference.boxes) > 0
    assert len(exported.boxes) == len(reference.boxes)
    reference_order = torch.argsort(reference.boxes.conf, descending=True)
    exported_order = torch.argsort(exported.boxes.conf, descending=True)
    assert torch.equal(
        exported.boxes.cls[exported_order], reference.boxes.cls[reference_order]
    )
    box_difference = (
        exported.boxes.xyxy[exported_order] - reference.boxes.xyxy[reference_order]
    )
    assert box_difference.abs().max() < YOLO_TOLERANCE_PIXELS
    return exported_order, reference_order


@pytest.mark.parametrize("builtin_engine", EXPORTED_MODEL_ENGINE_PACKAGES)
def test_yolo_on_a_builtin_engine_matches_yolo_on_pytorch(builtin_engine, portrait_rgb):
    require_engine(builtin_engine)
    reference, exported = detection_results(portrait_rgb, builtin_engine)

    assert_same_boxes(exported, reference)
    assert exported.names == reference.names
    assert exported.masks is None


@pytest.mark.parametrize("builtin_engine", EXPORTED_MODEL_ENGINE_PACKAGES)
def test_yolo_pose_on_a_builtin_engine_matches_yolo_pose_on_pytorch(
    builtin_engine, portrait_rgb
):
    require_engine(builtin_engine)
    reference, exported = pose_results(portrait_rgb, builtin_engine)

    exported_order, reference_order = assert_same_boxes(exported, reference)
    reference_keypoints = reference.keypoints[reference_order]
    visible = reference_keypoints.conf > VISIBLE_KEYPOINT_CONFIDENCE
    assert visible.any()
    keypoint_difference = (
        exported.keypoints[exported_order].xy[visible] - reference_keypoints.xy[visible]
    )
    assert keypoint_difference.abs().max() < YOLO_TOLERANCE_PIXELS


def test_tracking_on_an_exported_yolo_names_the_pytorch_engine(portrait_rgb):
    pytest.importorskip("ultralytics")
    from yolo import YOLOTransform

    element = YOLOTransform()
    element.set_property("track", True)
    element.set_property("model-name", YOLO_MODEL)
    element.set_property("engine-name", ENGINE_NAMES[1])
    element.do_load_model()

    with pytest.raises(ValueError, match="pytorch"):
        element.do_forward(portrait_rgb)


def zero_shot_detector(model_name, engine_name):
    from zeroshotdetector import ZeroShotDetector

    element = ZeroShotDetector()
    element.set_property("labels", ZERO_SHOT_LABELS)
    element.set_property("model-name", model_name)
    element.set_property("engine-name", engine_name)
    element.do_load_model()
    return element


def test_zero_shot_on_onnx_matches_zero_shot_on_pytorch(portrait_rgb):
    reference, exported = (
        zero_shot_detector(OWL_MODEL, engine_name).do_forward(portrait_rgb)
        for engine_name in ENGINE_NAMES
    )

    assert reference["boxes"]
    assert len(exported["boxes"]) == len(reference["boxes"])
    assert exported["labels"] == reference["labels"]
    assert exported["scores"] == pytest.approx(
        reference["scores"], abs=ZERO_SHOT_SCORE_TOLERANCE
    )
    box_difference = np.abs(np.array(exported["boxes"]) - np.array(reference["boxes"]))
    assert box_difference.max() < ZERO_SHOT_BOX_TOLERANCE_PIXELS


def test_a_model_with_text_in_its_image_encoder_does_not_export():
    with pytest.raises(ValueError, match="pytorch engine"):
        zero_shot_detector(GROUNDING_MODEL, ENGINE_NAMES[1])
