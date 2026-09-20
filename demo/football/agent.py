import asyncio
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import httpx
from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO = Path(__file__).resolve().parents[2]
RUN_SCRIPT = REPO / "demo/football/run.sh"
PYML_MCP = REPO / ".venv/bin/pyml-mcp"
LOG_DIR = Path(tempfile.gettempdir())

LLAMA_RUNTIME = Path.home() / ".local/share/liquid/runtime"
LLAMA_SERVER = LLAMA_RUNTIME / "llama-vulkan/llama-server"
MODEL = LLAMA_RUNTIME / "models/03b74727a860-Qwen3.5-9B-Q4_K_M.gguf"
LLAMA_URL = "http://127.0.0.1:8089"
GPU_LAYERS = 20
CPU_THREADS = 6
CONTEXT_TOKENS = 8192
MAX_TOKENS = 300
LLAMA_START_TIMEOUT_SECONDS = 90
CHAT_TIMEOUT_SECONDS = 120

MAX_TOOL_ROUNDS = 8
FAULT_CONFIDENCE = "0.99"
PASSTHROUGH_TOOLS = ("set_property", "get_property", "pipeline_status", "stop_pipeline")
START_TOOL = {
    "type": "function",
    "function": {
        "name": "start_football_demo",
        "description": "Start the football broadcast overlay pipeline on the demo video.",
        "parameters": {"type": "object", "properties": {}},
    },
}

SYSTEM_PROMPT = """You operate a live GStreamer football broadcast pipeline through tools.
The pipeline has two named elements you may change:
- detector (pyml_yolo): property `confidence`, float 0 to 1, default 0.1. The minimum score a detection needs. Set high, players and the ball stop being detected and their markers vanish.
- overlay (pyml_football_overlay): property `show-ball`, bool, default false, draws the marker on the ball. Property `trails`, bool, default false, draws a fading motion trail behind each player. Property `show-hud`, bool, default true, draws the focal-player HUD with headshot, contacts and distance.
Spell booleans as true or false and numbers plainly.
Start the pipeline with start_football_demo only when asked to start it.
For a plain request to change a property, call set_property straight away.
When the user reports something wrong with the picture, first call pipeline_status and then get_property on the properties above to find the cause. Only then change something, and say what was wrong.
Never change anything the user did not ask about. After the tools finish, answer in one short sentence."""


def start_llama_server():
    log = open(LOG_DIR / "football-agent-llama.log", "w")
    process = subprocess.Popen(
        [
            str(LLAMA_SERVER),
            "-m",
            str(MODEL),
            "-ngl",
            str(GPU_LAYERS),
            "-t",
            str(CPU_THREADS),
            "-c",
            str(CONTEXT_TOKENS),
            "--port",
            LLAMA_URL.rsplit(":", 1)[1],
            "--jinja",
            "--no-ui",
        ],
        stdout=log,
        stderr=log,
    )
    deadline = time.monotonic() + LLAMA_START_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SystemExit(f"llama-server exited, see {log.name}")
        try:
            httpx.get(f"{LLAMA_URL}/health").raise_for_status()
            return process
        except httpx.HTTPError:
            time.sleep(0.5)
    process.kill()
    raise SystemExit(
        f"llama-server not healthy after {LLAMA_START_TIMEOUT_SECONDS}s, see {log.name}"
    )


def pyml_mcp_transport():
    log = open(LOG_DIR / "football-agent-pyml-mcp.log", "w")
    parameters = StdioServerParameters(
        command=str(PYML_MCP), env=dict(os.environ), cwd=str(REPO)
    )
    return stdio_client(parameters, errlog=log)


def openai_tool(tool):
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


async def model_tools(mcp):
    listed = await mcp.list_tools()
    passthrough = [
        openai_tool(tool) for tool in listed.tools if tool.name in PASSTHROUGH_TOOLS
    ]
    return [START_TOOL, *passthrough]


def demo_pipeline():
    return subprocess.run(
        [str(RUN_SCRIPT), "print"], check=True, capture_output=True, text=True
    ).stdout.strip()


async def run_tool(mcp, name, arguments):
    if name == "start_football_demo":
        name, arguments = "start_pipeline", {"pipeline": demo_pipeline()}
    result = await mcp.call_tool(name, arguments)
    text = "\n".join(block.text for block in result.content if hasattr(block, "text"))
    return compact_json(text)


def compact_json(text):
    try:
        return json.dumps(json.loads(text))
    except ValueError:
        return text


def format_call(name, arguments):
    rendered = ", ".join(f"{key}={value!r}" for key, value in arguments.items())
    return f"{name}({rendered})"


async def chat(http, messages, tools):
    response = await http.post(
        f"{LLAMA_URL}/v1/chat/completions",
        json={
            "messages": messages,
            "tools": tools,
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=CHAT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]


async def answer(http, mcp, messages, tools, request):
    messages.append({"role": "user", "content": request})
    for _ in range(MAX_TOOL_ROUNDS):
        reply = await chat(http, messages, tools)
        calls = reply.get("tool_calls") or []
        messages.append(
            {
                "role": "assistant",
                "content": reply.get("content") or "",
                "tool_calls": calls,
            }
        )
        if not calls:
            print(reply.get("content") or "")
            return
        for call in calls:
            name = call["function"]["name"]
            arguments = json.loads(call["function"]["arguments"] or "{}")
            print(f"  -> {format_call(name, arguments)}")
            # a failed tool goes back to the model as text
            try:
                output = await run_tool(mcp, name, arguments)
            except Exception as error:
                output = f"error: {error}"
            print(f"  <- {output}")
            messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": output}
            )
    print("gave up after too many tool rounds")


async def inject_fault(mcp):
    output = await run_tool(
        mcp,
        "set_property",
        {"element": "detector", "property": "confidence", "value": FAULT_CONFIDENCE},
    )
    print(f"  (fault injected, model not told) {output}")


async def repl(http, mcp):
    tools = await model_tools(mcp)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print(
        "football agent ready. type a command, /fault to break the detector, /quit to exit"
    )
    while True:
        try:
            request = (await asyncio.to_thread(input, "> ")).strip()
        except EOFError:
            return
        if not request:
            continue
        if request == "/quit":
            return
        if request == "/fault":
            await inject_fault(mcp)
            continue
        await answer(http, mcp, messages, tools, request)


async def main():
    llama = start_llama_server()
    try:
        async with Client(pyml_mcp_transport()) as mcp, httpx.AsyncClient() as http:
            await repl(http, mcp)
    finally:
        llama.terminate()
        llama.wait()


if __name__ == "__main__":
    asyncio.run(main())
