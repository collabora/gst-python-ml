import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "plugins" / "python"))

from utils.format_converter import FormatConverter  # noqa: E402

HEIGHT, WIDTH = 2, 3
RED, GREEN, BLUE, ALPHA = 10, 20, 30, 255

PACKED_PIXELS = {
    "RGB": (RED, GREEN, BLUE),
    "BGR": (BLUE, GREEN, RED),
    "RGBA": (RED, GREEN, BLUE, ALPHA),
    "ARGB": (ALPHA, RED, GREEN, BLUE),
    "BGRA": (BLUE, GREEN, RED, ALPHA),
    "ABGR": (ALPHA, BLUE, GREEN, RED),
}


@pytest.mark.parametrize("format", PACKED_PIXELS)
def test_the_rgb_frame_is_a_view_with_channels_in_rgb_order(format):
    pixel = PACKED_PIXELS[format]
    data = np.tile(np.array(pixel, dtype=np.uint8), (HEIGHT, WIDTH, 1))

    rgb = FormatConverter.extract_rgb(data, format)

    assert rgb.shape == (HEIGHT, WIDTH, 3)
    assert np.all(rgb == (RED, GREEN, BLUE))
    assert np.shares_memory(rgb, data)


def test_a_channel_count_that_does_not_match_the_format_is_refused():
    with pytest.raises(ValueError):
        FormatConverter.extract_rgb(np.zeros((HEIGHT, WIDTH, 3), np.uint8), "RGBA")
