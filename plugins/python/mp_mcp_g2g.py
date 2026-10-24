# mp-mcp on g2g
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import json
import shlex
import subprocess
import sys
import threading

import mp_launch

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
    rewritten = " ".join(mp_launch.rewrite_for_g2g(tokens, shells))
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
    shells = mp_launch.element_shells()
    server = subprocess.Popen(
        [mp_launch.required_g2g_binary(), "--mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env=mp_launch.launch_environment(),
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
