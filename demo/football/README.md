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

## Environment knobs

| Var        | Default | Meaning |
|------------|---------|---------|
| `BACKEND`  | `pt`    | `pt` = PyTorch `pyml_yolo`; `fp16` = ONNX FP16 via `pyml_objectdetector` (CUDA). |
| `INTERVAL` | `3`     | Run detection every Nth frame; the tracker/overlay still update every frame, so it stays smooth at ~N× less inference cost. The main real-time lever. |

```bash
BACKEND=fp16 demo/football/run.sh display     # faster inference path
INTERVAL=5   demo/football/run.sh display     # detect every 5th frame
INTERVAL=1   demo/football/run.sh             # detect every frame (max accuracy)
```


