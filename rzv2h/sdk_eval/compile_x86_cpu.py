#!/usr/bin/env python3
# compile_x86_cpu.py — functional x86 (host CPU) compile of an ONNX model using
# the DRP-AI TVM (MERA fork) relay stack, for testing the MERA/TVM
# graph_executor runtime WITHOUT a board/NPU/QEMU.
#
# It mirrors the TVM-backend half of the SDK's compile_cpu_only_onnx_model.py
# but retargets to native "llvm" (host x86, default g++) and skips the DRP-AI
# pre-processing runtime (we preprocess in Python in drpai_engine). Output:
# <out>/deploy.{so,json,params} — loadable by tvm.contrib.graph_executor, i.e.
# by the drpai_runtime shim's TVM backend.
#
# Run inside the drpai-tvm-v2h container:
#   python3 compile_x86_cpu.py <model.onnx> <out_dir> [input_name] [C,H,W]
import os
import sys

import onnx
import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.build_module import build as _build, bind_params_by_name
from tvm.relay.param_dict import save_param_dict
from tvm.ir.transform import Sequential, PassContext

model_file = sys.argv[1]
out_dir = sys.argv[2]
input_name = sys.argv[3] if len(sys.argv) > 3 else "images"
chw = [int(x) for x in (sys.argv[4].split(",") if len(sys.argv) > 4 else [3, 640, 640])]
input_shape = [1] + chw

os.makedirs(out_dir, exist_ok=True)
print(f"[x86 compile] {model_file} input {input_name}={input_shape} -> {out_dir}")

onnx_model = onnx.load_model(model_file)
mod, params = relay.frontend.from_onnx(onnx_model, {input_name: input_shape})
if params:
    mod["main"] = bind_params_by_name(mod["main"], params)

with PassContext(opt_level=3):
    mod = Sequential([
        transform.SimplifyInference(),
        transform.FoldConstant(),
        transform.FoldExplicitPadding(),
        transform.BackwardFoldScaleAxis(),
        transform.ForwardFoldScaleAxis(),
        transform.FoldConstant(),
        transform.DynamicToStatic(),
        transform.RemoveUnusedFunctions(),
    ])(mod)

target = "llvm"  # native host (x86), no aarch64 cross target
with PassContext(opt_level=3):
    graph, lib, all_params = _build(mod, target=target, target_host=target, params=params)

lib.export_library(os.path.join(out_dir, "deploy.so"))  # default host compiler -> x86 .so
with open(os.path.join(out_dir, "deploy.json"), "w") as f:
    f.write(graph)
with open(os.path.join(out_dir, "deploy.params"), "wb") as f:
    f.write(save_param_dict(all_params))
print(f"[x86 compile finished] -> {out_dir}/deploy.so,deploy.json,deploy.params")
