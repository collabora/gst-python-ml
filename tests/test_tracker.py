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

from tracker import SortTracker, merge_covered_detections  # noqa: E402

FRAMES = 20
# a 40x60 player moving 30 px a frame keeps under 0.15 IoU with its last box
BOX_W, BOX_H, STEP = 40, 60, 30
LABEL = "player"


def detection(x, y, score=0.9, w=BOX_W, h=BOX_H):
    return [x, y, w, h, score, LABEL]


def ids_over_frames(tracker, frames):
    ids = set()
    for detections in frames:
        for track_id, _box, _label in tracker.update(detections):
            ids.add(track_id)
    return ids


def test_merge_drops_a_box_covered_by_a_higher_scoring_one():
    body = detection(100, 100, score=0.9)
    legs = detection(105, 130, score=0.4, w=30, h=30)
    beside = detection(150, 100, score=0.5)
    kept = merge_covered_detections([legs, body, beside], 0.5)
    assert kept == [body, beside]


def test_a_fast_small_box_keeps_one_id_through_the_distance_gate():
    frames = [[detection(100 + STEP * i, 200)] for i in range(FRAMES)]
    assert len(ids_over_frames(SortTracker(), frames)) == 1
    # with overlap only no track reaches three hits
    assert len(ids_over_frames(SortTracker(distance_gate=0), frames)) == 0


def test_a_second_box_on_a_matched_player_starts_no_track():
    frames = [[detection(100, 200, score=0.9)]]
    for i in range(1, FRAMES):
        x = 100 + 2 * i
        # a shadow box beside the player, overlapping it a little every frame
        frames.append([detection(x, 200, score=0.9), detection(x + 30, 210, score=0.6)])
    assert len(ids_over_frames(SortTracker(), frames)) == 1
    assert len(ids_over_frames(SortTracker(new_track_max_overlap=1.0), frames)) == 2


def test_two_separate_players_get_two_ids():
    frames = [[detection(100 + i, 100), detection(500 - i, 400)] for i in range(FRAMES)]
    assert len(ids_over_frames(SortTracker(), frames)) == 2
