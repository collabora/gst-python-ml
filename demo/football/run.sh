#!/usr/bin/env bash
# Football broadcast-overlay demo.
#
# Ppipeline:
#   detector  ->  pyml_tracker (ByteTrack)  ->  pyml_football_overlay
#
# Usage:
#   demo/football/run.sh [INPUT.mp4] [OUTPUT.mp4] [WxH]      # file -> annotated mp4
#   demo/football/run.sh display [INPUT.mp4] [WxH]           # file -> live on-screen
#   demo/football/run.sh camera [/dev/videoN] [WxH]          # live camera -> on-screen
#   demo/football/run.sh print [INPUT.mp4] [WxH]             # echo the display pipeline for pyml-mcp
#   HOST=g2g demo/football/run.sh ...                         # same, hosted on g2g-launch-py
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
source .venv/bin/activate
export GST_PLUGIN_PATH="$REPO/plugins:${GST_PLUGIN_PATH:-}"
# gst-launch's python loader runs the system interpreter, so hand it the venv.
VENV_SITE="$(python -c 'import site; print(":".join(site.getsitepackages()))')"
export PYTHONPATH="$REPO/plugins/python:$VENV_SITE:${PYTHONPATH:-}"

BACKEND="${BACKEND:-pt}"
HOST="${HOST:-gst}"        # gst = gst-launch-1.0, g2g = g2g-launch-py hosting the same python elements
# The weights live on the Hugging Face Hub; this is a no-op once cached.
python demo/football/fetch_models.py "$BACKEND" >&2
INTERVAL="${INTERVAL:-1}"   # run detection every Nth frame; the ball trail needs every frame
CONF="${CONF:-0.1}"        # detector confidence threshold (low = more detections)
IMGSZ="${IMGSZ:-640}"      # network input size; 1280 sees the ball far more often (pt backend only)
IOU="${IOU:-0.7}"          # NMS IoU (ultralytics/football_analyzer default)
NEWTRACK="${NEWTRACK:-0.25}" # min confidence to START a new track (ByteTrack gate; kills ghosts)
DRAWCONF="${DRAWCONF:-0}"  # min confidence to DRAW a detection (0 = draw all; raise to trim weak boxes)
MERGE="${MERGE:-0.5}"      # collapse overlapping boxes (lower=merge more; 0 disables) so one player=one circle
SMOOTH="${SMOOTH:-0.6}"    # temporal EMA on circle positions (0=off, higher=smoother but more lag)
TRAILS="${TRAILS:-false}"  # draw motion trails behind players
BALL_TRAIL="${BALL_TRAIL:-false}"
SHOW_BALL="${SHOW_BALL:-false}" # draw the ball marker
CLASSES="ball,goalkeeper,player,referee"
if [[ "$HOST" == "g2g" ]]; then
  [[ "$BACKEND" == "pt" ]] || { echo "HOST=g2g runs the pt backend only" >&2; exit 1; }
  DETECTOR_ELEMENT="pyelement module=yolo class=YOLOTransform"
  TRACKER_ELEMENT="pyelement module=tracker class=TrackerTransform"
  OVERLAY_ELEMENT="pyelement module=football_overlay class=FootballOverlay"
  CONVERT="videoconvertscale"
  DECODE="qtdemux ! ffmpegdec ! $CONVERT"
  DISPLAY_SINK="videoconvertscale ! waylandsink"
  CAMERA_SINK="videoconvertscale ! waylandsink"
  ENCODE="x264enc ! mp4mux"
  LAUNCH="g2g-launch-py"
else
  DETECTOR_ELEMENT="pyml_yolo name=detector"
  TRACKER_ELEMENT="pyml_tracker"
  OVERLAY_ELEMENT="pyml_football_overlay name=overlay"
  CONVERT="videoconvert ! videoscale"
  DECODE="decodebin ! $CONVERT"
  LAUNCH="gst-launch-1.0 -e"
fi
TRACK="$TRACKER_ELEMENT tracker-type=bytetrack new-track-confidence=$NEWTRACK"
# Detection-based overlay: circles sit on the raw per-frame detections (no
# tracking drift/phantoms/doubles); merge collapses overlaps and
# position-smoothing low-passes the positions. DRAWCONF defaults 0 so no
# detection is hidden; the tracker still runs so the HUD keeps its stats.
OVERLAY="$OVERLAY_ELEMENT class-names=$CLASSES team-colors=true trails=$TRAILS ball-trail=$BALL_TRAIL show-ball=$SHOW_BALL show-ids=false show-labels=false draw-from-detections=true min-confidence=$DRAWCONF merge-iou=$MERGE position-smoothing=$SMOOTH highlight-focal=false"

if [[ "$BACKEND" == "fp16" ]]; then
  # nvidia is a namespace package (no __file__), so walk __path__ for the pip CUDA libs.
  export LD_LIBRARY_PATH="$(python -c "import os,glob,nvidia;print(':'.join(sorted({d for p in nvidia.__path__ for d in glob.glob(os.path.join(p,'*','lib'))})))"):${LD_LIBRARY_PATH:-}"
  DETECT="pyml_objectdetector name=detector engine-name=onnx model-name=models/football/football_fp16.onnx device=cuda:0 input-format=nchw post-process=anchor_free interval=$INTERVAL"
  IN_FMT="RGB"
else
  DETECT="$DETECTOR_ELEMENT model-name=models/football/football device=cuda:0 interval=$INTERVAL imgsz=$IMGSZ confidence=$CONF nms-iou=$IOU"
  IN_FMT="RGBA"
fi

POST_DETECT="$TRACK"
[[ "$IN_FMT" == "RGB" ]] && POST_DETECT="$TRACK ! videoconvert ! video/x-raw,format=RGBA"

if [[ "$HOST" == "g2g" ]]; then
  # g2g has no queue element and needs none
  CHAIN="$DETECT ! $POST_DETECT ! $OVERLAY"
else
  # A queue at each stage boundary turns the serial chain into a threaded
  # pipeline: while inference runs on frame N, the sink renders N-1 and the
  # decoder reads N+1. Nothing is dropped (leaky=no, the default).
  Q="queue max-size-buffers=8 max-size-time=0 max-size-bytes=0"
  # no min-threshold: it blocks output whenever the level dips below it
  PREROLL="queue max-size-buffers=100 max-size-time=0 max-size-bytes=0"
  CHAIN="$Q ! $DETECT ! $Q ! $POST_DETECT ! $Q ! $OVERLAY"
  DISPLAY_SINK="$PREROLL ! videoconvert ! autovideosink sync=true"
  CAMERA_SINK="$Q ! videoconvert ! autovideosink sync=false"
  ENCODE="$Q ! videoconvert ! openh264enc ! h264parse ! mp4mux"
fi

MODE="${1:-file}"
if [[ "$MODE" == "camera" ]]; then
  DEV="${2:-/dev/video0}"; SIZE="${3:-1280x720}"
  W="${SIZE%x*}"; H="${SIZE#*x}"
  echo "[$HOST/$BACKEND] live camera $DEV @ ${W}x${H} -> display"
  exec $LAUNCH \
    v4l2src device="$DEV" ! $CONVERT \
    ! "video/x-raw,width=${W},height=${H},format=${IN_FMT}" \
    ! $CHAIN \
    ! $CAMERA_SINK
elif [[ "$MODE" == "display" || "$MODE" == "print" ]]; then
  IN="${2:-data/soccer_tracking.mp4}"
  SIZE="${3:-1280x720}"
  W="${SIZE%x*}"; H="${SIZE#*x}"
  [[ -f "$IN" ]] || { echo "input not found: $IN" >&2; exit 1; }
  PIPELINE="filesrc location=$IN ! $DECODE \
    ! video/x-raw,width=${W},height=${H},format=${IN_FMT} \
    ! $CHAIN \
    ! $DISPLAY_SINK"
  if [[ "$MODE" == "print" ]]; then
    echo "$PIPELINE"
    exit 0
  fi
  echo "[$HOST/$BACKEND] '$IN' @ ${W}x${H} -> live display (real-time)"
  exec $LAUNCH $PIPELINE
else
  IN="${1:-data/soccer_tracking.mp4}"
  OUT="${2:-demo/football/out.mp4}"
  SIZE="${3:-1280x720}"
  W="${SIZE%x*}"; H="${SIZE#*x}"
  [[ -f "$IN" ]] || { echo "input not found: $IN" >&2; exit 1; }
  echo "[$HOST/$BACKEND] '$IN' @ ${W}x${H} -> '$OUT'"
  $LAUNCH \
    filesrc location="$IN" ! $DECODE \
    ! "video/x-raw,width=${W},height=${H},format=${IN_FMT}" \
    ! $CHAIN \
    ! $ENCODE ! filesink location="$OUT"
  echo "Done: $OUT"
fi
