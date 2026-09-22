import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
PLUGIN_DIR = BASE_DIR / "plugins" / "python"

# the plugin loader reads GST_PLUGIN_PATH at Gst.init
os.environ["GST_PLUGIN_PATH"] = str(BASE_DIR / "plugins")
sys.path.insert(0, str(PLUGIN_DIR))

gi = pytest.importorskip("gi")
gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
from gi.repository import Gst  # noqa: E402

Gst.init(None)

from alert import AlertTransform  # noqa: E402

TOKEN_VARIABLE = "PYML_TEST_WEBHOOK_TOKEN"
TOKEN = "s3cret-value"
ALERT_PAYLOAD = {"timestamp": 0, "rule": {"class": "person"}, "detection": {}}
POST_TIMEOUT = 5


class RecordingHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.server.received_headers = dict(self.headers)
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.end_headers()
        self.server.posted.set()

    def log_message(self, *args):
        pass


@pytest.fixture
def webhook_server():
    server = HTTPServer(("127.0.0.1", 0), RecordingHandler)
    server.posted = threading.Event()
    server.received_headers = {}
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()
    server.server_close()


def post_one_alert(server, token_variable):
    element = AlertTransform()
    host, port = server.server_address
    element.set_property("webhook-url", f"http://{host}:{port}/alert")
    element.set_property("webhook-token-environment-variable", token_variable)
    element._send_webhook(ALERT_PAYLOAD)
    return element


def test_a_named_environment_variable_becomes_a_bearer_header(
    webhook_server, monkeypatch
):
    monkeypatch.setenv(TOKEN_VARIABLE, TOKEN)

    post_one_alert(webhook_server, TOKEN_VARIABLE)

    assert webhook_server.posted.wait(POST_TIMEOUT), "the webhook was never posted"
    assert webhook_server.received_headers["Authorization"] == f"Bearer {TOKEN}"


def test_the_webhook_stays_unauthenticated_when_no_variable_is_named(webhook_server):
    post_one_alert(webhook_server, "")

    assert webhook_server.posted.wait(POST_TIMEOUT), "the webhook was never posted"
    assert "Authorization" not in webhook_server.received_headers


def test_an_unset_variable_fails_instead_of_posting(webhook_server, monkeypatch):
    monkeypatch.delenv(TOKEN_VARIABLE, raising=False)

    with pytest.raises(ValueError) as caught:
        post_one_alert(webhook_server, TOKEN_VARIABLE)

    assert TOKEN_VARIABLE in str(caught.value)
    assert not webhook_server.posted.wait(1), "an unauthenticated alert was posted"
