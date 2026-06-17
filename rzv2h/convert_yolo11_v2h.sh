#!/usr/bin/env bash
# Compile YOLO11 (ONNX) -> RZ/V2H DRP-AI (INT8) deploy dir, using the REAL
# mera2 + DRP-AI Translator i8 + DRP-AI Quantizer flow.
#
# RUN INSIDE the drpai-tvm-v2h container (built via rzv2h/sdk_eval/build_image.sh),
# with this repo mounted at /work. RZ/V2H uses the DRP-AI INT8 accelerator, so
# quantization is MANDATORY and calibration images are required — this is why
# the plain FP compile_onnx_model.py does NOT work for V2H.
#
# Usage (inside container):
#   ./rzv2h/convert_yolo11_v2h.sh [MODEL.onnx] [OUT_DIR] [CALIB_DIR] [IMGSZ]
# Defaults assume the repo is at /work and the ONNX is exported already
# (e.g. `yolo export model=models/yolo11m/yolo11m.pt format=onnx imgsz=640` on a
# host with ultralytics — the container has no ultralytics).
set -euo pipefail

ONNX="${1:-/work/models/yolo11m/yolo11m.onnx}"
OUT="${2:-/work/rzv2h/yolo11m_drpai_v2h}"
CALIB="${3:-/work/rzv2h/calib}"
IMGSZ="${4:-640}"

: "${TVM_ROOT:?run inside the drpai-tvm-v2h container (TVM_ROOT unset)}"
export PRODUCT=V2H
export SDK="$(find /opt/ -name sysroots -type d | head -1)/../"
export TRANSLATOR="$(find /opt/ -name python_api -type d | head -1)/../../"
: "${QUANTIZER:?QUANTIZER env not set (expected from the image)}"
export PATH="$TVM_ROOT/tutorials:$PATH"          # so run_drp_compiler.sh resolves
chmod +x "$TVM_ROOT"/tutorials/*.sh 2>/dev/null || true   # SDK ships them non-+x

[[ -f "$ONNX" ]] || { echo "ONNX not found: $ONNX (export it first)"; exit 1; }
[[ -d "$CALIB" ]] || { echo "calibration image dir not found: $CALIB"; exit 1; }

# The stock quant script preprocesses calibration images as ImageNet (224 +
# mean/std) — wrong for YOLO (needs IMGSZ, /255, RGB, CHW). Patch that one line.
python3 - "$TVM_ROOT/tutorials/compile_onnx_model_quant.py" "$IMGSZ" <<'PYEOF'
import sys
p, sz = sys.argv[1], int(sys.argv[2])
s = open(p).read()
old = "input_data = pre_process_imagenet_pytorch(image, mean, stdev, need_transpose=True)"
new = ("input_data = (cv2.resize(image,(%d,%d))[:,:,::-1]"
       ".astype('float32')/255.0).transpose(2,0,1)" % (sz, sz))
if old in s:
    open(p, "w").write(s.replace(old, new)); print("[patch] calibration preprocessing ->", sz)
else:
    print("[patch] calibration line already patched / not found")
PYEOF

rm -rf "$OUT"
cd "$TVM_ROOT/tutorials"
python3 compile_onnx_model_quant.py "$ONNX" \
  -o "$OUT" -i images -s "1,3,${IMGSZ},${IMGSZ}" \
  -t "$SDK" -d "$TRANSLATOR" -c "$QUANTIZER" --images "$CALIB"

echo
echo "Done. RZ/V2H DRP-AI (INT8) deploy dir: $OUT"
echo "  sub_0000__CPU_DRP_TVM/{deploy.so,deploy.json,deploy.params}  (aarch64 + DRP-AI)"
echo "  preprocess/   (DRP-AI pre-processing runtime objects)"
echo "Copy $OUT to the board; load sub_0000__CPU_DRP_TVM with the MERA runtime."
