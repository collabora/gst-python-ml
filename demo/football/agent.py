import asyncio
import json
import os
import subprocess
import sys
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
# 4B fits on the GPU beside the detector.
MODEL = LLAMA_RUNTIME / "models/Qwen3.5-4B-Q4_K_M.gguf"
LLAMA_URL = "http://127.0.0.1:8089"
GPU_LAYERS = 99
CPU_THREADS = 6
CONTEXT_TOKENS = 8192
MAX_TOKENS = 300
LLAMA_START_TIMEOUT_SECONDS = 90
CHAT_TIMEOUT_SECONDS = 120

MAX_TOOL_ROUNDS = 8
FAULT_CONFIDENCE = "0.99"
# Jack sense reports this headset port unavailable.
HEADSET_SOURCE = "alsa_input.pci-0000_34_00.6.analog-stereo"
HEADSET_PORT = "analog-input-mic"
WHISPER_MODEL = "base"
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
- overlay (pyml_football_overlay): property `show-ball`, bool, default false, draws the ball marker. The ball track is that marker plus its trail, and the trail is drawn only when `trails` is also true. Property `trails`, bool, default false, draws a fading motion trail behind each player, and behind the ball when `show-ball` is true. Property `show-hud`, bool, default true, draws the focal-player HUD with headshot, contacts and distance.
Showing the ball track, or tracking the ball, sets `show-ball` to true and `trails` to true. Hiding or stopping the ball track sets `show-ball` to false and leaves `trails` as it is. Showing the player tracks sets `trails` to true and leaves `show-ball` as it is. Hiding the player tracks sets `trails` to false and leaves `show-ball` as it is.
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


DEMO_VIDEO = "data/iStock-1446288409.mp4"
RECORD_PATH = "/tmp/football-voice.mp4"


def demo_pipeline():
    pipeline = subprocess.run(
        [str(RUN_SCRIPT), "print", DEMO_VIDEO],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    sink = "autovideosink sync=true"
    record = (
        'textoverlay name=transcript text="" font-desc="Sans, 32" '
        "halignment=center valignment=bottom shaded-background=true "
        "auto-resize=false wrap-mode=word ! tee name=view "
        "view. ! queue max-size-buffers=8 max-size-time=0 max-size-bytes=0 ! "
        "videoconvert ! autovideosink name=display sync=true "
        "view. ! queue max-size-buffers=8 max-size-time=0 max-size-bytes=0 ! "
        "videoconvert ! video/x-raw,format=I420 ! openh264enc ! h264parse ! mux. "
        f"pulsesrc device={HEADSET_SOURCE} do-timestamp=true ! "
        "queue max-size-time=2000000000 max-size-buffers=0 max-size-bytes=0 ! "
        "audioconvert ! audioresample ! audio/x-raw,rate=48000,channels=1 ! "
        "avenc_aac ! mux. "
        "mp4mux name=mux fragment-duration=500 fragment-mode=first-moov-then-finalise ! "
        f"filesink name=record location={RECORD_PATH}"
    )
    if sink not in pipeline:
        raise SystemExit("football pipeline has no display sink for the transcript")
    return pipeline.replace(sink, record)


async def run_tool(mcp, name, arguments):
    if name == "start_football_demo":
        name, arguments = "start_pipeline", {
            "pipeline": demo_pipeline(),
            "loop": False,
        }
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


async def chat(http, messages, tools, max_tokens=MAX_TOKENS, timeout=CHAT_TIMEOUT_SECONDS):
    response = await http.post(
        f"{LLAMA_URL}/v1/chat/completions",
        json={
            "messages": messages,
            "tools": tools,
            "max_tokens": max_tokens,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]


async def answer(
    http, mcp, messages, tools, request, speak=True, max_tokens=MAX_TOKENS, timeout=CHAT_TIMEOUT_SECONDS
):
    messages.append({"role": "user", "content": request})
    for _ in range(MAX_TOOL_ROUNDS):
        reply = await chat(http, messages, tools, max_tokens, timeout)
        calls = reply.get("tool_calls") or []
        messages.append(
            {
                "role": "assistant",
                "content": reply.get("content") or "",
                "tool_calls": calls,
            }
        )
        if not calls:
            if speak:
                print(reply.get("content") or "")
            return
        for call in calls:
            name = call["function"]["name"]
            arguments = json.loads(call["function"]["arguments"] or "{}")
            print(f"  -> {format_call(name, arguments)}", flush=True)
            # a failed tool goes back to the model as text
            try:
                output = await run_tool(mcp, name, arguments)
            except Exception as error:
                output = f"error: {error}"
            print(f"  <- {output}", flush=True)
            messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": output}
            )
        if not speak:
            return
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


def use_headset_mic():
    subprocess.run(
        ["pactl", "set-source-port", HEADSET_SOURCE, HEADSET_PORT],
        check=True,
    )


def start_microphone(loop, queue):
    os.environ["GST_PLUGIN_PATH"] = os.pathsep.join(
        [str(REPO / "plugins"), os.environ.get("GST_PLUGIN_PATH", "")]
    )
    sys.path.insert(0, str(REPO / "plugins" / "python"))
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    use_headset_mic()
    pipeline = Gst.parse_launch(
        "pulsesrc name=mic ! audioconvert ! audioresample ! "
        "audio/x-raw,format=S16LE,rate=16000,channels=1 ! "
        f"pyml_whispertranscribe name=stt device=cpu language=en "
        f"model-name={WHISPER_MODEL} beam-size=1 ! "
        "appsink name=words emit-signals=true sync=false"
    )
    pipeline.get_by_name("mic").set_property("device", HEADSET_SOURCE)
    pipeline.get_by_name("stt").set_property(
        "initial-prompt",
        "show the ball track. hide the ball track. start the football demo.",
    )

    def on_sample(sink):
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK
        buf = sample.get_buffer()
        ok, info = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.ERROR
        text = bytes(info.data).decode("utf-8", errors="replace").strip()
        buf.unmap(info)
        if text:
            loop.call_soon_threadsafe(queue.put_nowait, text)
        return Gst.FlowReturn.OK

    pipeline.get_by_name("words").connect("new-sample", on_sample)
    if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
        raise SystemExit("whisper pipeline failed to start")
    return pipeline


def whisper_error(pipeline):
    from gi.repository import Gst

    bus = pipeline.get_bus()
    found = None
    while True:
        message = bus.pop()
        if message is None:
            return found
        if message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            found = err.message if not debug else f"{err.message} ({debug})"


async def voice(http, mcp):
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    # A misheard phrase must not be able to stop the picture.
    tools = [
        tool
        for tool in await model_tools(mcp)
        if tool["function"]["name"] != "stop_pipeline"
    ]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    queue = asyncio.Queue()
    pipeline = start_microphone(asyncio.get_running_loop(), queue)
    print(await run_tool(mcp, "start_football_demo", {}), flush=True)
    print(
        "listening on the headset mic. whisper base is on the cpu. "
        "say a command, Ctrl-C to stop.",
        flush=True,
    )
    try:
        while True:
            try:
                text = await asyncio.wait_for(queue.get(), timeout=0.5)
            except TimeoutError:
                error = whisper_error(pipeline)
                if error:
                    raise SystemExit(f"whisper: {error}")
                continue
            while True:
                try:
                    text = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            # A long chat was teaching the model to keep the ball track on.
            messages[:] = [{"role": "system", "content": SYSTEM_PROMPT}]
            print(f"> {text}", flush=True)
            try:
                await run_tool(
                    mcp,
                    "set_property",
                    {"element": "transcript", "property": "text", "value": text},
                )
            except Exception as error:
                print(f"  ! transcript: {error}", flush=True)
            try:
                await answer(
                    http,
                    mcp,
                    messages,
                    tools,
                    text,
                    speak=False,
                    max_tokens=160,
                    timeout=20,
                )
            except Exception as error:
                print(f"  ! {error}", flush=True)
    finally:
        pipeline.set_state(Gst.State.NULL)


async def main():
    llama = start_llama_server()
    try:
        async with Client(pyml_mcp_transport()) as mcp, httpx.AsyncClient() as http:
            if "--voice" in sys.argv:
                await voice(http, mcp)
            else:
                await repl(http, mcp)
    finally:
        llama.terminate()
        llama.wait()


if __name__ == "__main__":
    asyncio.run(main())
