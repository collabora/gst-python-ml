import sys
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent / "plugins" / "python"
sys.path.insert(0, str(PLUGIN_DIR))

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from engine.optical_flow_engine import bilinear_sample_by_gather  # noqa: E402

# a raft correlation volume: one channel per pixel of the first frame
VOLUME_SHAPE = (30, 1, 23, 40)
NEIGHBOURHOOD_SHAPE = (30, 7, 7, 2)
# reaches past every edge
COORDINATE_SPAN = torch.tensor([60.0, 35.0])
COORDINATE_OFFSET = 10.0
SAMPLE_TOLERANCE = 1e-5


def test_gather_sampler_matches_the_grid_sample_raft_calls():
    from torchvision.models.optical_flow._utils import grid_sample

    generator = torch.Generator().manual_seed(0)
    volume = torch.rand(VOLUME_SHAPE, generator=generator)
    coordinates = (
        torch.rand(NEIGHBOURHOOD_SHAPE, generator=generator) * COORDINATE_SPAN
        - COORDINATE_OFFSET
    )

    reference = grid_sample(volume, coordinates, align_corners=True, mode="bilinear")
    sampled = bilinear_sample_by_gather(
        volume, coordinates, align_corners=True, mode="bilinear"
    )

    assert sampled.shape == reference.shape
    assert (sampled - reference).abs().max() < SAMPLE_TOLERANCE
