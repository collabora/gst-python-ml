#!/usr/bin/env python3
# x86_runtime_check.py — run a real input through the MERA/TVM graph_executor
# (via the drpai_runtime shim's TVM backend) and check parity against the
# known-good ONNX output. Run INSIDE the drpai-tvm-v2h container; needs the
# x86 deploy dir + the pre-saved input/onnx-reference .npy files.
import sys
import numpy as np

sys.path.insert(0, "/work/rzv2h/emulation")   # drpai_runtime shim (TVM backend)
sys.path.insert(0, "/work/plugins/python")     # utils.detection_decoder (pure numpy)

import drpai_runtime
from utils.detection_decoder import decode

DEPLOY = "/work/rzv2h/yolo11m_x86_cpu"
x = np.load("/work/rzv2h/_x86test_input.npy").astype(np.float32)
ref = np.load("/work/rzv2h/_x86test_onnxout.npy").astype(np.float32).reshape(-1)

rt = drpai_runtime.Runtime()
assert rt.load(DEPLOY), "drpai_runtime.load failed"
rt.set_input(0, x)
rt.run()
out = np.asarray(rt.get_output(0), dtype=np.float32).reshape(-1)

n = min(out.size, ref.size)
maxdiff = float(np.max(np.abs(out[:n] - ref[:n]))) if n else float("nan")
print(f"TVM out size={out.size} ref size={ref.size} max|TVM-ONNX|={maxdiff:.3e}")

tvm_det = decode(out.reshape(1, 84, 8400), "anchor_free")[0]
onnx_det = decode(ref.reshape(1, 84, 8400), "anchor_free")[0]
print(f"detections  TVM={len(tvm_det['boxes'])}  ONNX={len(onnx_det['boxes'])}")
if len(tvm_det["boxes"]):
    print("TVM labels:", sorted(set(int(c) for c in tvm_det["labels"])))
print("PASS" if maxdiff < 1e-2 and len(tvm_det["boxes"]) == len(onnx_det["boxes"]) else "CHECK")
