"""Unit tests for `pyml-launch`'s gst -> g2g pipeline rewrite.

These run the real `element_shells` scan over the real plugin directory, so a
plugin that stops declaring `register_gst_element` fails a test rather than
silently dropping out of the map.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "plugins" / "python"))

import pyml_launch  # noqa: E402


@pytest.fixture(scope="module")
def shells():
    return pyml_launch.element_shells()


def test_element_shells_finds_the_hosted_elements(shells):
    assert shells["pyml_yolo"] == ("yolo", "YOLOTransform")
    assert shells["pyml_objectdetector"][0] == "objectdetector"
    assert shells["pyml_overlay"] == ("overlay", "Overlay")


def test_hosted_element_becomes_pyelement_keeping_its_properties(shells):
    assert (
        pyml_launch.rewrite_segment(
            "pyml_yolo model-name=yolo11m device=cuda:0", shells
        )
        == "pyelement module=yolo class=YOLOTransform model-name=yolo11m device=cuda:0"
    )


def test_overlay_becomes_the_native_element(shells):
    assert pyml_launch.rewrite_segment("pyml_overlay", shells) == "analyticsoverlay"


def test_overlay_properties_are_refused_rather_than_guessed(shells):
    with pytest.raises(SystemExit, match="show-track"):
        pyml_launch.rewrite_segment("pyml_overlay tracking=True", shells)


def test_unknown_pyml_element_is_refused(shells):
    with pytest.raises(SystemExit, match="pyml_nonesuch"):
        pyml_launch.rewrite_segment("pyml_nonesuch", shells)


def test_shared_elements_pass_through_untouched(shells):
    for segment in ("filesrc location=data/people.mp4", "decodebin", "videoscale"):
        assert pyml_launch.rewrite_segment(segment, shells) == segment


def test_raw_video_caps_gain_a_format_only_when_they_lack_one(shells):
    assert (
        pyml_launch.rewrite_segment("video/x-raw,width=640,height=480", shells)
        == "video/x-raw,width=640,height=480,format=RGBA"
    )
    pinned = "video/x-raw,format=NV12,width=640,height=480"
    assert pyml_launch.rewrite_segment(pinned, shells) == pinned


def test_whole_readme_pipeline_rewrites(shells):
    pipeline = (
        "filesrc location=data/soccer_tracking.mp4 ! decodebin ! videoconvert "
        "! videoscale ! video/x-raw,width=640,height=480 "
        "! pyml_yolo model-name=yolo11m device=cuda:0 track=True "
        "! pyml_overlay ! videoconvert ! autovideosink"
    )
    assert pyml_launch.rewrite_for_g2g(pipeline, shells) == (
        "filesrc location=data/soccer_tracking.mp4 ! decodebin ! videoconvert "
        "! videoscale ! video/x-raw,width=640,height=480,format=RGBA "
        "! pyelement module=yolo class=YOLOTransform model-name=yolo11m "
        "device=cuda:0 track=True "
        "! analyticsoverlay ! videoconvert ! autovideosink"
    )
