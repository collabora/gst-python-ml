import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

BASE_DIRECTORY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIRECTORY / "plugins" / "python"))

from engine.engine_factory import EngineFactory  # noqa: E402


class FakeVART:
    input_shapes = [[1, 4, 6, 3]]
    input_shape_formats = ["NHWC"]
    input_types = ["uint8"]
    instances = []

    def __init__(self, snapshot_dir, network_name, output_names=None, npu_only=False):
        self.snapshot_dir = snapshot_dir
        self.network_name = network_name
        self.output_names = output_names
        self.npu_only = npu_only
        self.execute_calls = []
        self.instances.append(self)

    def get_input_shapes(self):
        return self.input_shapes

    def get_input_shape_formats(self):
        return self.input_shape_formats

    def get_input_types(self):
        return self.input_types

    def execute(self, inputs):
        self.execute_calls.append(inputs)
        return [np.array([[1.0, 2.0]], dtype=np.float32)]


@pytest.fixture
def fake_vart_runtime(monkeypatch):
    FakeVART.input_shapes = [[1, 4, 6, 3]]
    FakeVART.input_shape_formats = ["NHWC"]
    FakeVART.input_types = ["uint8"]
    FakeVART.instances = []
    runner = ModuleType("runner")
    runner.VART = FakeVART
    monkeypatch.setitem(sys.modules, "runner", runner)
    return FakeVART


def test_factory_creates_vart_engine_without_importing_the_target_runtime():
    engine = EngineFactory.create("vart")
    assert engine.__class__.__name__ == "VARTEngine"


def test_loads_snapshot_with_directory_name_as_network(fake_vart_runtime, tmp_path):
    snapshot_path = tmp_path / "resnet50"
    snapshot_path.mkdir()
    engine = EngineFactory.create("vart")
    engine.do_set_device("npu")

    assert engine.do_load_model(str(snapshot_path)) is True
    runtime = fake_vart_runtime.instances[-1]
    assert runtime.snapshot_dir == str(snapshot_path)
    assert runtime.network_name == "resnet50"
    assert runtime.npu_only is False
    assert engine.model is runtime


def test_loads_named_network_and_selected_outputs(fake_vart_runtime, tmp_path):
    snapshot_path = tmp_path / "snapshot"
    snapshot_path.mkdir()
    engine = EngineFactory.create("vart")
    engine.do_set_device("npu-only")

    engine.do_load_model(
        str(snapshot_path), network_name="detector", output_names="boxes, scores"
    )

    runtime = fake_vart_runtime.instances[-1]
    assert runtime.network_name == "detector"
    assert runtime.output_names == ["boxes", "scores"]
    assert runtime.npu_only is True


def test_loads_network_name_from_model_name(fake_vart_runtime, tmp_path):
    snapshot_path = tmp_path / "snapshot.NPU.resnet50.TF"
    snapshot_path.mkdir()
    engine = EngineFactory.create("vart")
    engine.do_set_device("npu")

    engine.do_load_model(f"{snapshot_path}::resnet50")

    runtime = fake_vart_runtime.instances[-1]
    assert runtime.snapshot_dir == str(snapshot_path)
    assert runtime.network_name == "resnet50"


def test_runs_nhwc_batch_with_runtime_input_type(fake_vart_runtime, tmp_path):
    snapshot_path = tmp_path / "resnet50"
    snapshot_path.mkdir()
    engine = EngineFactory.create("vart")
    engine.do_set_device("npu")
    engine.do_load_model(str(snapshot_path))
    engine.post_process = "none"
    frames = np.arange(144, dtype=np.float32).reshape(2, 4, 6, 3)

    result = engine.do_forward(frames)

    runtime = fake_vart_runtime.instances[-1]
    input_array = runtime.execute_calls[-1][0]
    assert input_array.shape == (2, 4, 6, 3)
    assert input_array.dtype == np.uint8
    np.testing.assert_array_equal(input_array, frames.astype(np.uint8))
    np.testing.assert_array_equal(result, [[1.0, 2.0]])


def test_uses_runtime_nchw_format_for_one_frame(fake_vart_runtime, tmp_path):
    fake_vart_runtime.input_shapes = [[1, 3, 4, 6]]
    fake_vart_runtime.input_shape_formats = ["NCHW"]
    snapshot_path = tmp_path / "resnet50"
    snapshot_path.mkdir()
    engine = EngineFactory.create("vart")
    engine.do_set_device("npu")
    engine.do_load_model(str(snapshot_path))
    engine.post_process = "none"
    frame = np.zeros((4, 6, 3), dtype=np.uint8)

    engine.do_forward(frame)

    runtime = fake_vart_runtime.instances[-1]
    assert runtime.execute_calls[-1][0].shape == (1, 3, 4, 6)


def test_refuses_snapshot_with_multiple_inputs(fake_vart_runtime, tmp_path):
    fake_vart_runtime.input_shapes = [[1, 4, 6, 3], [1, 2]]
    snapshot_path = tmp_path / "network"
    snapshot_path.mkdir()
    engine = EngineFactory.create("vart")
    engine.do_set_device("npu")

    with pytest.raises(ValueError, match="requires one input, got 2"):
        engine.do_load_model(str(snapshot_path))


def test_refuses_non_snapshot_path(fake_vart_runtime, tmp_path):
    engine = EngineFactory.create("vart")
    engine.do_set_device("npu")

    with pytest.raises(FileNotFoundError, match="requires a snapshot directory"):
        engine.do_load_model(str(tmp_path / "missing"))


def test_refuses_unknown_device():
    engine = EngineFactory.create("vart")

    with pytest.raises(ValueError, match="VART has no device=fpga"):
        engine.do_set_device("fpga")
