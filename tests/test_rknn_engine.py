import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

BASE_DIRECTORY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIRECTORY / "plugins" / "python"))

from engine.engine_factory import EngineFactory  # noqa: E402


class FakeRKNNLite:
    NPU_CORE_AUTO = "auto"
    NPU_CORE_0 = "core-0"
    NPU_CORE_1 = "core-1"
    NPU_CORE_2 = "core-2"
    NPU_CORE_ALL = "all"
    load_status = 0
    initialization_status = 0
    instances = []

    def __init__(self):
        self.loaded_model = None
        self.core_mask = None
        self.inference_calls = []
        self.released = False
        self.instances.append(self)

    def load_rknn(self, model_name):
        self.loaded_model = model_name
        return self.load_status

    def init_runtime(self, core_mask=None):
        self.core_mask = core_mask
        return self.initialization_status

    def inference(self, inputs, data_format):
        self.inference_calls.append((inputs, data_format))
        return [np.array([[1.0, 2.0]], dtype=np.float32)]

    def release(self):
        self.released = True


@pytest.fixture
def fake_rknn_runtime(monkeypatch):
    FakeRKNNLite.load_status = 0
    FakeRKNNLite.initialization_status = 0
    FakeRKNNLite.instances = []
    package = ModuleType("rknnlite")
    api = ModuleType("rknnlite.api")
    api.RKNNLite = FakeRKNNLite
    package.api = api
    monkeypatch.setitem(sys.modules, "rknnlite", package)
    monkeypatch.setitem(sys.modules, "rknnlite.api", api)
    return FakeRKNNLite


def test_factory_creates_rknn_engine_without_importing_the_board_runtime():
    engine = EngineFactory.create("rknn")
    assert engine.__class__.__name__ == "RKNNEngine"


@pytest.mark.parametrize(
    ("device", "core_mask"),
    [
        ("npu", None),
        ("rknn", None),
        ("npu:0", "core-0"),
        ("npu:1", "core-1"),
        ("npu:2", "core-2"),
        ("npu:all", "all"),
    ],
)
def test_loads_model_on_selected_npu_core(
    fake_rknn_runtime, tmp_path, device, core_mask
):
    model_path = tmp_path / "model.rknn"
    model_path.touch()
    engine = EngineFactory.create("rknn")
    engine.do_set_device(device)

    assert engine.do_load_model(str(model_path)) is True
    runtime = fake_rknn_runtime.instances[-1]
    assert runtime.loaded_model == str(model_path)
    assert runtime.core_mask == core_mask
    assert engine.model is runtime


def test_runs_single_nhwc_frame_without_changing_its_values(
    fake_rknn_runtime, tmp_path
):
    model_path = tmp_path / "model.rknn"
    model_path.touch()
    engine = EngineFactory.create("rknn")
    engine.do_set_device("npu")
    engine.do_load_model(str(model_path))
    engine.post_process = "none"
    frame = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)

    result = engine.do_forward(frame)

    runtime = fake_rknn_runtime.instances[-1]
    inputs, data_format = runtime.inference_calls[-1]
    assert data_format == ["nhwc"]
    assert inputs[0].shape == (1, 2, 4, 3)
    np.testing.assert_array_equal(inputs[0][0], frame)
    np.testing.assert_array_equal(result, [[1.0, 2.0]])


def test_runs_nchw_frames_one_at_a_time(fake_rknn_runtime, tmp_path):
    model_path = tmp_path / "model.rknn"
    model_path.touch()
    engine = EngineFactory.create("rknn")
    engine.do_set_device("npu")
    engine.do_load_model(str(model_path))
    engine.input_format = "nchw"
    engine.post_process = "none"
    frames = np.zeros((2, 3, 4, 3), dtype=np.uint8)

    results = engine.do_forward(frames)

    runtime = fake_rknn_runtime.instances[-1]
    assert len(runtime.inference_calls) == 2
    inputs, data_format = runtime.inference_calls[0]
    assert data_format == ["nchw"]
    assert inputs[0].shape == (1, 3, 3, 4)
    assert len(results) == 2


def test_releases_runtime_when_npu_initialization_fails(fake_rknn_runtime, tmp_path):
    fake_rknn_runtime.initialization_status = -1
    model_path = tmp_path / "model.rknn"
    model_path.touch()
    engine = EngineFactory.create("rknn")
    engine.do_set_device("npu")

    with pytest.raises(RuntimeError, match="status -1"):
        engine.do_load_model(str(model_path))

    assert fake_rknn_runtime.instances[-1].released is True
    assert engine.model is None


def test_refuses_non_rknn_model(fake_rknn_runtime, tmp_path):
    model_path = tmp_path / "model.onnx"
    model_path.touch()
    engine = EngineFactory.create("rknn")
    engine.do_set_device("npu")

    with pytest.raises(FileNotFoundError, match="requires a .rknn model"):
        engine.do_load_model(str(model_path))
