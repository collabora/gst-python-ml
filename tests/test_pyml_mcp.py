import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "plugins" / "python"))

pytest.importorskip("mcp")
pytest.importorskip("gi")

import pyml_mcp  # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402

pytestmark = pytest.mark.skipif(
    not any(element["name"] == "pyml_metasink" for element in pyml_mcp.list_elements()),
    reason="the python plugin loader did not register the pyml elements",
)

STATUS_POLLS = 100
POLL_SECONDS = 0.1


def wait_until_ended():
    import time

    for _ in range(STATUS_POLLS):
        if pyml_mcp.pipeline_status()["ended"]:
            return
        time.sleep(POLL_SECONDS)
    pytest.fail(f"pipeline did not end: {pyml_mcp.pipeline_status()}")


def test_text_flows_from_the_sink_to_latest_metadata(tmp_path):
    text = tmp_path / "hello.txt"
    text.write_text("hello")
    pyml_mcp.start_pipeline(
        f"filesrc location={text} ! text/x-raw,format=utf8 ! pyml_metasink"
    )
    wait_until_ended()
    assert [record["text"] for record in pyml_mcp.latest_metadata()] == ["hello"]
    assert pyml_mcp.stop_pipeline() == {"state": "none"}


def test_a_property_can_change_while_the_pipeline_runs():
    pyml_mcp.start_pipeline(
        "videotestsrc name=source pattern=snow ! pyml_metasink name=sink"
    )
    assert pyml_mcp.get_property("source", "pattern") == "snow"
    assert pyml_mcp.set_property("source", "pattern", "ball") == {"pattern": "ball"}
    assert pyml_mcp.set_property("sink", "location", "a b") == {"location": "a b"}
    with pytest.raises(ToolError, match="no property"):
        pyml_mcp.set_property("source", "no-such-property", "1")
    pyml_mcp.stop_pipeline()


def test_a_sink_without_a_location_is_kept_off_stdout():
    pyml_mcp.start_pipeline("videotestsrc ! pyml_metasink name=sink")
    assert pyml_mcp.get_property("sink", "location") == os.devnull
    pyml_mcp.stop_pipeline()


def test_a_bad_description_is_reported_and_leaves_nothing_running():
    with pytest.raises(ToolError, match="nosuchelement"):
        pyml_mcp.start_pipeline("nosuchelement ! fakesink")
    assert pyml_mcp.pipeline_status() == {"state": "none"}


def test_inspect_lists_the_sink_location():
    described = pyml_mcp.inspect("pyml_metasink")
    defaults = {spec["name"]: spec["default"] for spec in described["properties"]}
    assert defaults["location"] == ""
    assert defaults["last-sample"] == ""
