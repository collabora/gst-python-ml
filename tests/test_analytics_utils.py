import os
import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
PLUGIN_DIR = BASE_DIR / "plugins" / "python"

os.environ["GST_PLUGIN_PATH"] = str(BASE_DIR / "plugins")
sys.path.insert(0, str(PLUGIN_DIR))

gi = pytest.importorskip("gi")
gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

Gst.init(None)

from utils.analytics_utils import AnalyticsUtils  # noqa: E402


@pytest.mark.parametrize(
    ("full_label", "expected"),
    [
        ("stream_0_person_id_7", (7, "person")),
        ("stream_1_ball_id_2", (2, "ball")),
        ("stream_0_sports ball_id_3", (3, "sports ball")),
        ("stream_0_id_5", (5, "id_5")),
        ("stream_0_dog", (None, "dog")),
        ("whatever", (None, "whatever")),
    ],
)
def test_extract_id_from_label(full_label, expected):
    assert AnalyticsUtils().extract_id_from_label(full_label) == expected
