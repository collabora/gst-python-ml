import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "plugins" / "python"))

import pyml_launch  # noqa: E402
import pyml_mcp_g2g  # noqa: E402

GST_SPELLED_PIPELINE = (
    "videotestsrc num-buffers=2 ! videoconvert ! mp_tracker name=tracker ! fakesink"
)
MP_MCP_ENTRY_POINT = "import modelpipe; modelpipe.mcp()"
SERVER_TIMEOUT_SECONDS = 120


@pytest.fixture(scope="module")
def shells():
    return pyml_launch.element_shells()


def tool_call(request_id, name, arguments):
    params = {"name": name, "arguments": arguments}
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": params,
    }


def relayed(requests, shells):
    server_input = io.StringIO()
    replies = io.StringIO()
    lines = [json.dumps(request) + "\n" for request in requests]
    pyml_mcp_g2g.relay_requests(
        lines, server_input, pyml_mcp_g2g.LineWriter(replies), shells
    )
    forwarded = [json.loads(line) for line in server_input.getvalue().splitlines()]
    answered = [json.loads(line) for line in replies.getvalue().splitlines()]
    return forwarded, answered


def test_a_gst_spelled_pipeline_reaches_g2g_rewritten(shells):
    request = tool_call(1, "start_pipeline", {"pipeline": GST_SPELLED_PIPELINE})
    forwarded, answered = relayed([request], shells)
    assert answered == []
    assert forwarded[0]["params"]["arguments"]["pipeline"] == (
        "videotestsrc num-buffers=2 ! videoconvert ! pyelement module=tracker "
        "class=TrackerTransform name=tracker ! fakesink"
    )


def test_an_inserted_element_reaches_g2g_rewritten(shells):
    arguments = {"element": "mp_tracker", "target": "videoconvert0"}
    forwarded, _ = relayed([tool_call(1, "insert_transform", arguments)], shells)
    assert forwarded[0]["params"]["arguments"] == {
        "element": "pyelement module=tracker class=TrackerTransform",
        "target": "videoconvert0",
    }


def test_a_call_with_no_launch_line_passes_unchanged(shells):
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        tool_call(2, "set_property", {"element": "tracker", "property": "mp_x"}),
    ]
    forwarded, _ = relayed(requests, shells)
    assert forwarded == requests


def test_a_line_the_rewrite_refuses_is_answered_without_reaching_g2g(shells):
    pipeline = "videotestsrc ! mp_alert broker=localhost ! fakesink"
    forwarded, answered = relayed(
        [tool_call(7, "validate", {"pipeline": pipeline})], shells
    )
    assert forwarded == []
    assert answered[0]["id"] == 7
    assert answered[0]["error"]["code"] == pyml_mcp_g2g.INVALID_PARAMS
    assert "broker" in answered[0]["error"]["message"]


def test_pyml_mcp_on_g2g_runs_a_gst_spelled_pipeline():
    if not pyml_launch.g2g_binary():
        pytest.skip("needs a g2g-launch-py build")
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        tool_call(2, "launch", {"pipeline": GST_SPELLED_PIPELINE}),
        tool_call(
            3, "validate", {"pipeline": "videotestsrc ! mp_tracker x=1 ! fakesink"}
        ),
    ]
    server = subprocess.run(
        [sys.executable, "-c", MP_MCP_ENTRY_POINT],
        input="".join(json.dumps(request) + "\n" for request in requests),
        capture_output=True,
        text=True,
        timeout=SERVER_TIMEOUT_SECONDS,
        env={**os.environ, "MP_BACKEND": "g2g"},
    )
    replies = {
        reply["id"]: reply
        for reply in map(json.loads, server.stdout.splitlines())
        if "id" in reply
    }
    launched = json.loads(replies[2]["result"]["content"][0]["text"])
    validated = json.loads(replies[3]["result"]["content"][0]["text"])
    assert launched["ok"], launched
    assert not validated["ok"]
    assert "no property named 'x'" in validated["error"]
