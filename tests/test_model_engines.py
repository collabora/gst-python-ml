import sys
import tomllib
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "plugins" / "python"))

from engine import model_engines  # noqa: E402
from engine.support_matrix import (  # noqa: E402
    ENGINE_DEVICES,
    ENGINE_EXTRAS,
    ENGINE_PACKAGES,
    FILE_ENGINES,
    NCNN_BATCH_BROADCAST,
    UNTESTED_VARIANT_NOTE,
)

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
CAUSAL_LM_MODEL = "Qwen/Qwen3-0.6B"
CLIP_MODEL = "openai/clip-vit-base-patch32"
GROUNDING_MODEL = "IDEA-Research/grounding-dino-tiny"
TORCHVISION_CLASSIFIERS = ("resnet18", "mobilenet_v3_small")
TORCHVISION_RESNETS = ("resnet18",)


@pytest.fixture
def hub_configs(monkeypatch):
    architectures = {
        DEPTH_MODEL: ["DepthAnythingForDepthEstimation"],
        CAUSAL_LM_MODEL: ["Qwen3ForCausalLM"],
        CLIP_MODEL: ["CLIPModel"],
        GROUNDING_MODEL: ["GroundingDinoForObjectDetection"],
    }
    monkeypatch.setattr(model_engines, "hub_architectures", architectures.get)


@pytest.fixture(autouse=True)
def torchvision_names(monkeypatch):
    monkeypatch.setattr(
        model_engines,
        "is_torchvision_classifier",
        lambda name: name in TORCHVISION_CLASSIFIERS,
    )
    monkeypatch.setattr(
        model_engines, "is_torchvision_resnet", lambda name: name in TORCHVISION_RESNETS
    )


def tasks_of(model):
    return {task["task"]: task for task in model_engines.engine_options(model)["tasks"]}


def engine_of(task, engine):
    return next(entry for entry in task["engines"] if entry["engine"] == engine)


def test_a_depth_model_runs_on_onnx_and_ncnn_refuses_it(hub_configs):
    depth = tasks_of(DEPTH_MODEL)["depth"]
    assert engine_of(depth, "onnx")["status"] == model_engines.STATUS_RUNS
    ncnn = engine_of(depth, "ncnn")
    assert ncnn["status"] == model_engines.STATUS_REFUSED
    assert ncnn["reason"] == NCNN_BATCH_BROADCAST


def test_clip_serves_clip_and_embedding(hub_configs):
    assert set(tasks_of(CLIP_MODEL)) == {"clip", "embedding"}


def test_a_causal_lm_runs_on_llamacpp_and_not_on_tvm(hub_configs):
    llm = tasks_of(CAUSAL_LM_MODEL)["llm"]
    assert engine_of(llm, "llamacpp")["status"] == model_engines.STATUS_RUNS
    assert engine_of(llm, "tvm")["status"] == model_engines.STATUS_NOT_PORTED


def test_grounding_dino_runs_on_pytorch_only(hub_configs):
    zero_shot = tasks_of(GROUNDING_MODEL)["zero_shot"]
    assert zero_shot["note"] == "its image encoder takes the text"
    assert [entry["engine"] for entry in zero_shot["engines"]] == ["pytorch"]
    assert zero_shot["engines"][0]["status"] == model_engines.STATUS_PYTORCH_ONLY


def test_an_ultralytics_pose_name_resolves_to_pose():
    assert [resolved.task for resolved in model_engines.resolve("yolo11n-pose")] == [
        "pose"
    ]


def test_an_untested_ultralytics_variant_says_so():
    (resolved,) = model_engines.resolve("yolo11n-seg")
    assert resolved.task == "yolo"
    assert resolved.note == UNTESTED_VARIANT_NOTE


def test_an_onnx_file_lists_its_engines_as_untested(tmp_path):
    (task,) = model_engines.engine_options(str(tmp_path / "model.onnx"))["tasks"]
    assert {entry["engine"] for entry in task["engines"]} == set(FILE_ENGINES[".onnx"])
    assert {entry["status"] for entry in task["engines"]} == {
        model_engines.STATUS_UNTESTED
    }
    assert model_engines.MISSING_FILE_NOTE in task["note"]


def test_a_torchvision_resnet_serves_classifier_and_anomaly():
    assert set(tasks_of("resnet18")) == {"classifier", "anomaly"}


def test_an_unknown_name_resolves_to_nothing():
    assert model_engines.resolve("not-a-model") == []


def test_every_engine_table_names_a_known_engine():
    for table in (ENGINE_DEVICES, ENGINE_EXTRAS):
        assert set(table) == set(ENGINE_PACKAGES)
    for engines in FILE_ENGINES.values():
        assert set(engines) <= set(ENGINE_PACKAGES)


def test_every_engine_extra_is_a_pyproject_extra():
    with open(BASE_DIR / "pyproject.toml", "rb") as pyproject:
        extras = tomllib.load(pyproject)["project"]["optional-dependencies"]
    for engine_extras in ENGINE_EXTRAS.values():
        assert set(engine_extras) <= set(extras)
