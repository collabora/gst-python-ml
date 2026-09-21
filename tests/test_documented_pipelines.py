import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "plugins" / "python"))

from documented_pipelines import pipelines_by_section  # noqa: E402

PIPELINES_PATH = Path(__file__).resolve().parent.parent / "PIPELINES.md"
LEAST_OBJECT_DETECTION_PIPELINES = 10


def test_every_documented_pipeline_lands_under_its_heading():
    sections = pipelines_by_section(PIPELINES_PATH)
    assert sections
    assert len(sections["Object Detection"]) > LEAST_OBJECT_DETECTION_PIPELINES


def test_a_section_holds_descriptions_start_pipeline_can_take():
    for descriptions in pipelines_by_section(PIPELINES_PATH).values():
        for description in descriptions:
            assert not description.startswith("python")
            assert "!" in description or len(description.split()) == 1


def test_a_heading_claims_the_pipelines_below_it(tmp_path):
    doc = tmp_path / "PIPELINES.md"
    doc.write_text(
        "## First\n\n"
        "```\npython pyml-launch.py videotestsrc ! fakesink\n```\n\n"
        "### Second\n\n"
        "`python pyml-launch.py audiotestsrc ! fakesink`\n\n"
    )
    assert pipelines_by_section(doc) == {
        "First": ["videotestsrc ! fakesink"],
        "Second": ["audiotestsrc ! fakesink"],
    }
