# pyml-mcp on g2g
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Library General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Library General Public License for more details.
#
# You should have received a copy of the GNU Library General Public
# License along with this library; if not, write to the
# Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
# Boston, MA 02110-1301, USA.

import json
import shlex
import subprocess
import sys
import threading

import pyml_launch

LAUNCH_LINE_ARGUMENTS = {
    "validate": "pipeline",
    "launch": "pipeline",
    "start_pipeline": "pipeline",
    "validate_insertion": "element",
    "insert_transform": "element",
    "insert_chain": "chain",
    "add_branch": "branch",
}

INVALID_PARAMS = -32602


class LineWriter:
    def __init__(self, stream):
        self.stream = stream
        self.lock = threading.Lock()

    def write(self, line):
        with self.lock:
            self.stream.write(line)
            self.stream.flush()


def g2g_spelled(request, shells):
    if request.get("method") != "tools/call":
        return request
    params = request["params"]
    arguments = params.get("arguments", {})
    argument = LAUNCH_LINE_ARGUMENTS.get(params["name"])
    if argument not in arguments:
        return request
    tokens = shlex.split(arguments[argument])
    rewritten = " ".join(pyml_launch.rewrite_for_g2g(tokens, shells))
    arguments = {**arguments, argument: rewritten}
    return {**request, "params": {**params, "arguments": arguments}}


def refusal_reply(request, refusal):
    error = {"code": INVALID_PARAMS, "message": str(refusal)}
    return json.dumps({"jsonrpc": "2.0", "id": request.get("id"), "error": error})


def relay_requests(requests, server_input, replies, shells):
    for line in requests:
        if not line.strip():
            continue
        request = json.loads(line)
        # rewrite_for_g2g refuses a line by raising SystemExit
        try:
            request = g2g_spelled(request, shells)
        except (SystemExit, ValueError) as refusal:
            replies.write(refusal_reply(request, refusal) + "\n")
            continue
        server_input.write(json.dumps(request) + "\n")
        server_input.flush()


def relay_replies(server_output, replies):
    for line in server_output:
        replies.write(line)


def main():
    shells = pyml_launch.element_shells()
    server = subprocess.Popen(
        [pyml_launch.required_g2g_binary(), "--mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env=pyml_launch.launch_environment(),
    )
    replies = LineWriter(sys.stdout)
    reply_thread = threading.Thread(
        target=relay_replies, args=(server.stdout, replies), daemon=True
    )
    reply_thread.start()
    relay_requests(sys.stdin, server.stdin, replies, shells)
    server.stdin.close()
    reply_thread.join()
    return server.wait()
