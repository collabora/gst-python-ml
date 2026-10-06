# Football demo

Real-time football broadcast overlay: **detection → tracking → overlay**
(`pyml_yolo`/`pyml_objectdetector` -> `pyml_tracker` -> `pyml_football_overlay`).

The overlay draws a foot ellipse per player coloured by team (red/blue, voted
from jersey hue), a gold ellipse for referees, motion trails (off by default),
and a focal-player HUD with headshot, ball contacts, and distance travelled.
Players whose team isn't decided yet (and unclassifiable kits, e.g. the
goalkeeper) are left unmarked rather than drawn in a placeholder colour. The
ball is tracked for contact counting but its marker is off by default.

## Models

The detector weights (`football.pt`, `football.onnx`, `football_fp16.onnx`,
`football_int8.onnx`) are hosted on the Hugging Face Hub at
`boxerab/gst-python-ml-football`, not in git. `run.sh` downloads the one its
`BACKEND` needs into `models/football/` on first use. To fetch by hand, or to
use weights you already have, put them in that directory. Files already there
are not downloaded again.

```bash
python demo/football/fetch_models.py          # pt + fp16
python demo/football/fetch_models.py all
```

## Local setup

`run.sh` expects the repo venv at `.venv` and the system `gst-launch-1.0` with
the gst-python loader. It puts `plugins/` on `GST_PLUGIN_PATH` and the venv's
site-packages on `PYTHONPATH` itself, so nothing needs exporting first. On an
NVIDIA host `decodebin` picks `nvh264dec` and the detector runs on `cuda:0`.

```bash
cd gst-python-ml
uv sync
demo/football/run.sh display          # data/soccer_tracking.mp4, on screen
```

## Run

```bash
# file -> annotated MP4
demo/football/run.sh
demo/football/run.sh 08fd33_4.mp4 demo/football/out.mp4 1280x720

# file -> live on-screen
demo/football/run.sh display
demo/football/run.sh display 08fd33_4.mp4 1280x720

# live camera -> on-screen
demo/football/run.sh camera /dev/video0
```

## Typed commands through pyml-mcp

`agent.py` runs the display pipeline inside `pyml-mcp` and lets you change it
by typing. It starts the local model described below, spawns
`.venv/bin/pyml-mcp`, and shows each tool call as it happens. The model sees
five tools: `start_football_demo`, `set_property`, `get_property`,
`pipeline_status`, `stop_pipeline`. It knows `overlay show-ball`,
`overlay trails`, `overlay show-hud` and `detector confidence`.

```bash
.venv/bin/python demo/football/agent.py
> start the football demo
> show the ball
> /fault          # sets detector confidence to 0.99 without telling the model
> the circles on the players disappeared, what happened?
> /quit
```

`run.sh print` echoes the display pipeline the agent starts. Server logs go to
`football-agent-llama.log` and `football-agent-pyml-mcp.log` in the temp dir.
The voice log, when you use `--voice`, is `football-voice.log` in that same dir.

## Voice demo

```bash
export DISPLAY=:0 WAYLAND_DISPLAY=wayland-0 XDG_RUNTIME_DIR=/run/user/1000
.venv/bin/python demo/football/agent.py --voice
```

This machine is Fedora on Wayland (`wayland-0`), with Xwayland on `:0`. An
X11 screen grab of the picture comes out black. The window is `autovideosink`.

One process starts three pieces, all local, no network:

1. Whisper, a GStreamer pipeline in the agent.
2. Qwen, `llama-server` on `127.0.0.1:8089`.
3. The football pipeline, inside `pyml-mcp`, which the model drives with tools.

The clip is `data/iStock-1446288409.mp4` (31 s, 1920×1080, no audio track),
shown at 1280×720. `start_football_demo` is rewritten to
`start_pipeline` with that clip and `loop=False`.

### Whisper

`pulsesrc` captures the headset, then `pyml_whispertranscribe`:

- Model `base`, device `cpu`, `beam-size=1`, language `en`.
- Audio is S16LE, 16 kHz, mono.
- `initial-prompt` is `show the ball track. hide the ball track. start the football demo.`
- The element property is `beam_size` (gst name `beam-size`), default 5. Voice sets 1.
- CPU int8. Leave it on the CPU. The GPU is for the detector and Qwen.
- A short phrase is about 3 seconds after you stop talking, plus the 300 ms
  silence that ends an utterance.
- `tiny` was faster and misheard commands. Do not switch back to it for this demo.

Before the pipeline starts, the agent runs:

```bash
pactl set-source-port alsa_input.pci-0000_34_00.6.analog-stereo analog-input-mic
```

That source is the Ryzen HD Audio / ALC287 headset mic. Jack sense reports
`analog-input-mic` unavailable until the port is selected. Without it, Pulse
uses the C920 webcam mic. `pulsesrc` is pinned to that source by name.

Each transcript is written on the picture by `textoverlay name=transcript`
along the bottom, then sent to Qwen. If several transcripts are waiting, only
the newest one is sent.

### Qwen

Qwen 3.5 4B, Q4_K_M, file:

`~/.local/share/liquid/runtime/models/Qwen3.5-4B-Q4_K_M.gguf`

Server binary: `~/.local/share/liquid/runtime/llama-vulkan/llama-server`

```text
llama-server -m <that file> -ngl 99 -t 6 -c 8192 --port 8089 --jinja --no-ui
```

All layers sit on the RTX 3060 Laptop (6 GB) beside the detector, which uses
about 600 MB. A tool call is about 1 to 2 seconds once the prompt is cached.
Thinking is off (`chat_template_kwargs.enable_thinking=false`). Voice asks for
at most 160 tokens and gives the call 20 seconds. The spoken reply is not
generated. The tool call is the result.

The 9B weights are still on disk
(`models/03b74727a860-Qwen3.5-9B-Q4_K_M.gguf`) but this demo does not load
them. With the detector on the same card only 20 layers fit, and a tool call
took 5 to 7 seconds.

Voice does not keep the chat. Every phrase is a new request with the system
prompt only. A long history was making the 4B repeat "turn the ball track on"
and, past about 5,000 tokens, a closed connection left the listener stuck so
later phrases never ran.

`stop_pipeline` is hidden from the voice model. A misheard "go back" had
stopped the picture. The typed REPL still has that tool.

Property rules in the system prompt:

- "track the ball" / "show the ball track": `overlay show-ball=true` and `trails=true`. The ball trail is drawn only when both are true.
- "stop" / "hide" the ball track: `show-ball=false`, `trails` unchanged.
- "show player tracks": `trails=true`, `show-ball` unchanged.
- "hide player tracks": `trails=false`, `show-ball` unchanged.

### pyml-mcp

`.venv/bin/pyml-mcp` serves `plugins/python/pyml_mcp` over stdio. The agent
is the client. Tools the model can call are listed above. `start_pipeline`
takes a gst-launch string and `loop`. The voice demo passes `loop=False`.

`loop=True` arms one flush SEGMENT seek once the duration is known, then on
`SEGMENT_DONE` or EOS seeks again without flush. A flush seek at the end of
the clip stalls the picture. The seek runs on a side thread, not in the bus
sync handler, because that handler is on the streaming thread. The bus sync
handler drops every message. `tests/test_pyml_mcp.py` covers the loop with a
short `videotestsrc`.

`stop_pipeline` sends EOS and waits up to 8 seconds when the pipeline has an
element named `record`, then goes to NULL. EOS is what lets `mp4mux` finish
the file. NULL alone leaves an unplayable mp4.

### Recording

Voice also tees the overlay to `openh264enc` and mixes the headset
(`pulsesrc` on the same source Whisper uses) as AAC. The file is
`/tmp/football-voice.mp4` (`fragment-duration=500`, so it can be probed while
it is still open). The clip has no audio, so the soundtrack is the mic.

Do not turn `loop` back on while that filesink is in the pipeline. A seek
flushes the mux and the file never finalizes. The clip plays once. The live
mic does not send EOS, so the file keeps growing after the picture ends until
`stop_pipeline` runs.

Say the command and the words show on the picture a few seconds later. Qwen
then changes `overlay` or `detector`. From the end of the phrase to the
change is about 4 to 5 seconds: Whisper about 3, Qwen about 1 to 2.

## Environment knobs

| Var        | Default | Meaning |
|------------|---------|---------|
| `BACKEND`  | `pt`    | `pt` = PyTorch `pyml_yolo`; `fp16` = ONNX FP16 via `pyml_objectdetector` (CUDA). |
| `HOST`     | `gst`   | `gst` runs `gst-launch-1.0`. `g2g` runs the same three Python elements hosted by `g2g-launch-py` (needs it on `PATH`, `pt` backend only). |
| `INTERVAL` | `1`     | Run detection every Nth frame; the tracker/overlay still update every frame, so it stays smooth at ~N× less inference cost. The main real-time lever. The ball trail needs 1. |
| `IMGSZ`    | `640`   | Network input size for the `pt` backend. `1280` finds the ball on about half the frames instead of a third, at three times the inference cost. |
| `TRAILS`   | `false` | Draw motion trails behind the players and the ball. |
| `SHOW_BALL` | `false` | Draw the ball marker. |

```bash
BACKEND=fp16 demo/football/run.sh display     # faster inference path
INTERVAL=5   demo/football/run.sh display     # detect every 5th frame
INTERVAL=3   demo/football/run.sh             # detect every 3rd frame (cheaper, sparse ball trail)
TRAILS=true SHOW_BALL=true demo/football/run.sh display data/soccer_tracking.mp4
HOST=g2g demo/football/run.sh display              # same pipeline on the g2g host
```


