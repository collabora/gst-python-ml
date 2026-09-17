import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "benchmarks"))

pytest.importorskip("gi")

from engine_parity import (  # noqa: E402
    IOU_THRESHOLD,
    intersection_over_union,
    matched_box_fraction,
)


def box(x, y, width, height):
    return {"x": x, "y": y, "w": width, "h": height}


def test_the_same_box_overlaps_itself_completely():
    assert intersection_over_union(box(10, 10, 20, 20), box(10, 10, 20, 20)) == 1.0


def test_boxes_that_do_not_touch_do_not_overlap():
    assert intersection_over_union(box(0, 0, 10, 10), box(20, 20, 10, 10)) == 0.0


def test_boxes_sharing_only_an_edge_do_not_overlap():
    assert intersection_over_union(box(0, 0, 10, 10), box(10, 0, 10, 10)) == 0.0


def test_half_overlapping_boxes_share_a_third_of_their_area():
    assert intersection_over_union(
        box(0, 0, 10, 10), box(5, 0, 10, 10)
    ) == pytest.approx(1 / 3)


def test_every_reference_box_matches_a_shifted_copy():
    reference = {0.0: [box(0, 0, 10, 10)], 1.0: [box(50, 50, 10, 10)]}
    candidate = {0.0: [box(1, 0, 10, 10)], 1.0: [box(50, 51, 10, 10)]}
    assert matched_box_fraction(reference, candidate, IOU_THRESHOLD) == 1.0


def test_a_box_at_another_time_does_not_match():
    reference = {0.0: [box(0, 0, 10, 10)], 1.0: [box(0, 0, 10, 10)]}
    candidate = {0.0: [box(0, 0, 10, 10)]}
    assert matched_box_fraction(reference, candidate, IOU_THRESHOLD) == 0.5


def test_one_candidate_box_matches_only_one_reference_box():
    reference = {0.0: [box(0, 0, 10, 10), box(1, 1, 10, 10)]}
    candidate = {0.0: [box(0, 0, 10, 10)]}
    assert matched_box_fraction(reference, candidate, IOU_THRESHOLD) == 0.5


def test_a_box_below_the_threshold_does_not_match():
    reference = {0.0: [box(0, 0, 10, 10)]}
    candidate = {0.0: [box(6, 0, 10, 10)]}
    assert matched_box_fraction(reference, candidate, IOU_THRESHOLD) == 0.0


def test_no_reference_boxes_gives_no_fraction():
    assert matched_box_fraction({}, {0.0: [box(0, 0, 10, 10)]}, IOU_THRESHOLD) is None
