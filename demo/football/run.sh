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
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
source .venv/bin/activate
export GST_PLUGIN_PATH="$REPO/plugins:${GST_PLUGIN_PATH:-}"

BACKEND="${BACKEND:-pt}"
INTERVAL="${INTERVAL:-3}"   # run detection every Nth frame; tracker/overlay stay per-frame
CONF="${CONF:-0.1}"        # detector confidence threshold (low = more detections)
IOU="${IOU:-0.7}"          # NMS IoU (ultralytics/football_analyzer default)
NEWTRACK="${NEWTRACK:-0.25}" # min confidence to START a new track (ByteTrack gate; kills ghosts)
DRAWCONF="${DRAWCONF:-0}"  # min confidence to DRAW a detection (0 = draw all; raise to trim weak boxes)
MERGE="${MERGE:-0.5}"      # collapse overlapping boxes (lower=merge more; 0 disables) so one player=one circle
SMOOTH="${SMOOTH:-0.6}"    # temporal EMA on circle positions (0=off, higher=smoother but more lag)
CLASSES="ball,goalkeeper,player,referee"
TRACK="pyml_tracker tracker-type=bytetrack new-track-confidence=$NEWTRACK"
# Detection-based overlay: circles sit on the raw per-frame detections (no
# tracking drift/phantoms/doubles); merge collapses overlaps and
# position-smoothing low-passes the positions. DRAWCONF defaults 0 so no
# detection is hidden; the tracker still runs so the HUD keeps its stats.
OVERLAY="pyml_football_overlay class-names=$CLASSES team-colors=true trails=false show-ids=false show-labels=false draw-from-detections=true min-confidence=$DRAWCONF merge-iou=$MERGE position-smoothing=$SMOOTH highlight-focal=false"

if [[ "$BACKEND" == "fp16" ]]; then
  export LD_LIBRARY_PATH="$(python -c "import os,nvidia,glob;b=os.path.dirname(nvidia.__file__);print(':'.join(sorted(set(glob.glob(b+'/*/lib')))))"):${LD_LIBRARY_PATH:-}"
  DETECT="pyml_objectdetector engine-name=onnx model-name=models/football/football_fp16.onnx device=cuda:0 input-format=nchw post-process=anchor_free interval=$INTERVAL"
  IN_FMT="RGB"; FORCE_SQUARE=1
else
  DETECT="pyml_yolo model-name=models/football/football device=cuda:0 interval=$INTERVAL confidence=$CONF nms-iou=$IOU"
  IN_FMT="RGBA"; FORCE_SQUARE=0
fi

POST_DETECT="$TRACK"
[[ "$IN_FMT" == "RGB" ]] && POST_DETECT="$TRACK ! videoconvert ! video/x-raw,format=RGBA"

# A queue at each stage boundary turns the serial chain into a threaded
# pipeline: while inference runs on frame N, the sink renders N-1 and the
# decoder reads N+1. Nothing is dropped (leaky=no, the default).
Q="queue max-size-buffers=8 max-size-time=0 max-size-bytes=0"
# Pre-roll buffer before the display sink: build a head start of processed
# frames so real-time playback (sync=true) rides out per-frame inference
# jitter without stuttering. Smooths jitter, not a sustained throughput
# deficit -- if inference can't keep up on average, playback just lags
# (still no drops). Lower INTERVAL/raise the head start if it falls behind.
PREROLL="queue max-size-buffers=600 max-size-time=0 max-size-bytes=0 min-threshold-buffers=30"

# detector -> tracker -> overlay, with a thread boundary at each hop.
CHAIN="$Q ! $DETECT ! $Q ! $POST_DETECT ! $Q ! $OVERLAY"

MODE="${1:-file}"
if [[ "$MODE" == "camera" ]]; then
  DEV="${2:-/dev/video0}"; SIZE="${3:-1280x720}"
  [[ "$FORCE_SQUARE" == "1" ]] && SIZE="640x640"
  W="${SIZE%x*}"; H="${SIZE#*x}"
  echo "[$BACKEND] live camera $DEV @ ${W}x${H} -> autovideosink (needs a display)"
  exec gst-launch-1.0 -e \
    v4l2src device="$DEV" ! videoconvert ! videoscale \
    ! "video/x-raw,width=${W},height=${H},format=${IN_FMT}" \
    ! $CHAIN \
    ! $Q ! videoconvert ! autovideosink sync=false
elif [[ "$MODE" == "display" ]]; then
  IN="${2:-data/soccer_tracking.mp4}"
  SIZE="${3:-1280x720}"
  [[ "$FORCE_SQUARE" == "1" ]] && SIZE="640x640"
  W="${SIZE%x*}"; H="${SIZE#*x}"
  [[ -f "$IN" ]] || { echo "input not found: $IN" >&2; exit 1; }
  echo "[$BACKEND] '$IN' @ ${W}x${H} -> live display (real-time, sync=true)"
  exec gst-launch-1.0 -e \
    filesrc location="$IN" ! decodebin ! videoconvert ! videoscale \
    ! "video/x-raw,width=${W},height=${H},format=${IN_FMT}" \
    ! $CHAIN \
    ! $PREROLL ! videoconvert ! autovideosink sync=true
else
  IN="${1:-data/soccer_tracking.mp4}"
  OUT="${2:-demo/football/out.mp4}"
  SIZE="${3:-1280x720}"
  [[ "$FORCE_SQUARE" == "1" ]] && SIZE="640x640"
  W="${SIZE%x*}"; H="${SIZE#*x}"
  [[ -f "$IN" ]] || { echo "input not found: $IN" >&2; exit 1; }
  echo "[$BACKEND] '$IN' @ ${W}x${H} -> '$OUT'"
  gst-launch-1.0 -e \
    filesrc location="$IN" ! decodebin ! videoconvert ! videoscale \
    ! "video/x-raw,width=${W},height=${H},format=${IN_FMT}" \
    ! $CHAIN \
    ! $Q ! videoconvert ! openh264enc ! h264parse ! mp4mux ! filesink location="$OUT"
  echo "Done: $OUT"
fi
