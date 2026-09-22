import sys
from pathlib import Path

import numpy as np
import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent / "plugins" / "python"
sys.path.insert(0, str(PLUGIN_DIR))

torch = pytest.importorskip("torch")

from engine.super_res_engine import CHECKPOINT_URLS, SuperResEngine  # noqa: E402

STUB_SCALE = 2


# stands in for the spandrel descriptor
class DoublingUpsampler:
    def __call__(self, tensor):
        assert tensor.ndim == 4 and tensor.shape[1] == 3, tensor.shape
        assert tensor.dtype == torch.float32, tensor.dtype
        assert 0.0 <= float(tensor.min()) and float(tensor.max()) <= 1.0
        return tensor.repeat_interleave(STUB_SCALE, dim=2).repeat_interleave(
            STUB_SCALE, dim=3
        )


def engine_with_stub():
    engine = SuperResEngine()
    engine.device = "cpu"
    engine.upsampler = DoublingUpsampler()
    return engine


# a channel swap survives a pipeline run, so pin the colours to distinct values
FRAME = np.zeros((4, 6, 3), dtype=np.uint8)
FRAME[:, :, 0] = 10
FRAME[:, :, 1] = 120
FRAME[:, :, 2] = 250


def test_a_single_frame_keeps_its_channel_order_and_dtype():
    upscaled = engine_with_stub().do_forward(FRAME)

    assert upscaled.shape == (8, 12, 3)
    assert upscaled.dtype == np.uint8
    assert list(upscaled[0, 0]) == [10, 120, 250]


def test_a_batch_comes_back_as_a_list_of_frames():
    batch = np.stack([FRAME, FRAME[::-1]])

    upscaled = engine_with_stub().do_forward(batch)

    assert isinstance(upscaled, list) and len(upscaled) == 2
    assert all(frame.shape == (8, 12, 3) for frame in upscaled)


def test_a_read_only_frame_is_accepted():
    frame = FRAME.copy()
    frame.flags.writeable = False

    assert engine_with_stub().do_forward(frame).shape == (8, 12, 3)


def test_an_unknown_model_name_names_the_ones_that_work():
    with pytest.raises(ValueError) as caught:
        SuperResEngine().do_load_model("real-esrgan-x3")

    for name in CHECKPOINT_URLS:
        assert name in str(caught.value)
