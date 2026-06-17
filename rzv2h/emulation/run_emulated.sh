#!/usr/bin/env bash
# Run the DRP-AI object-detection pipeline on the DEV BOX using the emulated
# drpai_runtime (CPU/ONNX Runtime stand-in) — same engine code as the board,
# but no NPU. For validating the integration before deploying to RZ/V2H.
#
# Usage: ./run_emulated.sh [INPUT.mp4] [OUTPUT.mp4]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"

source .venv/bin/activate
export GST_PLUGIN_PATH="$REPO/plugins:${GST_PLUGIN_PATH:-}"
export PYTHONPATH="$HERE:${PYTHONPATH:-}"          # resolves `import drpai_runtime` to the fake

IN="${1:-08fd33_4.mp4}"
OUT="${2:-${IN%.*}_drpai_emu.mp4}"
DEPLOY="$HERE/yolo11m_drpai_v2h_emu"               # dir containing yolo11m.onnx

if [[ ! -f "$DEPLOY/yolo11m.onnx" ]]; then
  echo "Missing $DEPLOY/yolo11m.onnx — export it first:" >&2
  echo "  yolo export model=yolo11m.pt format=onnx imgsz=640 opset=12 simplify=True" >&2
  echo "  mkdir -p $DEPLOY && cp yolo11m.onnx $DEPLOY/" >&2
  exit 1
fi

echo "EMULATED DRP-AI run: '$IN' -> '$OUT' (CPU/ONNX, not the NPU)"
gst-launch-1.0 -e \
  filesrc location="$IN" ! decodebin ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! pyml_objectdetector engine-name=drpai model-name="$DEPLOY" device=drpai \
        input-format=nchw post-process=anchor_free \
  ! pyml_tracker tracker-type=bytetrack \
  ! videoconvert ! "video/x-raw,format=RGBA" \
  ! pyml_football_overlay show-ids=false show-labels=false \
  ! videoconvert ! openh264enc ! h264parse ! mp4mux ! filesink location="$OUT"
echo "Done: $OUT"
