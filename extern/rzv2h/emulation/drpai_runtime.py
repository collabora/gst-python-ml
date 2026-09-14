# drpai_runtime.py  —  off-board stand-in for the native pybind `drpai_runtime`.
# Copyright (C) 2024-2026 Collabora Ltd. — LGPL (see COPYING).
#
# Same interface as the C++ binding (Runtime.load / set_input / run /
# num_output / get_output), with two backends auto-selected by what's in the
# model directory and what's importable:
#
#   1. MERA / TVM graph_executor  — if the dir has deploy.so/json/params AND a
#      `tvm` runtime is importable (i.e. inside the Renesas DRP-AI TVM SDK
#      container, or on the board). This runs the REAL MERA/TVM runtime — the
#      faithful "test through the TVM runtime". On the board the deploy.so runs
#      on the DRP-AI NPU / Arm CPU; in the SDK container it runs on whatever the
#      module was compiled for (aarch64 needs QEMU; an x86-target build runs
#      natively for functional check).
#
#   2. ONNX Runtime (CPU)         — fallback look-alike for plain x86 dev boxes
#      with no SDK: runs the same yolo11m.onnx that feeds the DRP-AI compiler so
#      the engine's preprocess/reshape/decode path is exercised. Validates our
#      code, NOT the DRP-AI/MERA runtime.
#
# get_output() always returns a FLAT array, matching the C++ GetOutput buffer,
# so the engine's reshape-to-(1, 4+nc, anchors) path is genuinely tested.

import glob
import os

import numpy as np


class Runtime:
    def __init__(self):
        self._backend = None
        # tvm backend
        self._mod = None
        self._dev = None
        self._input_name = os.getenv("DRPAI_INPUT_NAME", "images")
        # onnx backend
        self._sess = None
        self._ort_input = None
        self._feed = None
        self._outputs = None

    def load(self, model_dir):
        deploy_so = os.path.join(model_dir, "deploy.so")
        if os.path.isfile(deploy_so) and self._try_load_tvm(model_dir, deploy_so):
            return True
        return self._try_load_onnx(model_dir)

    # ---- backend 1: real MERA / TVM graph_executor ----
    def _try_load_tvm(self, model_dir, deploy_so):
        try:
            import tvm
            from tvm.contrib import graph_executor
        except ImportError:
            return False
        try:
            lib = tvm.runtime.load_module(deploy_so)
            with open(os.path.join(model_dir, "deploy.json")) as f:
                graph = f.read()
            self._dev = tvm.cpu(0)
            self._mod = graph_executor.create(graph, lib, self._dev)
            with open(os.path.join(model_dir, "deploy.params"), "rb") as f:
                self._mod.load_params(bytearray(f.read()))
            self._backend = "tvm"
            print(
                f"[drpai_runtime] MERA/TVM graph_executor backend "
                f"(deploy.so, input='{self._input_name}') — real runtime"
            )
            return True
        except Exception as e:
            print(f"[drpai_runtime] TVM backend load failed ({e}); trying ONNX")
            return False

    # ---- backend 2: ONNX Runtime look-alike ----
    def _try_load_onnx(self, model_dir):
        try:
            import onnxruntime as ort
        except ImportError:
            print("[drpai_runtime] no TVM and no onnxruntime — cannot load")
            return False
        onnx_files = sorted(glob.glob(os.path.join(model_dir, "*.onnx")))
        if not onnx_files:
            print(f"[drpai_runtime] no deploy.so and no .onnx in {model_dir!r}")
            return False
        self._sess = ort.InferenceSession(
            onnx_files[0], providers=["CPUExecutionProvider"]
        )
        self._ort_input = self._sess.get_inputs()[0].name
        self._backend = "onnx"
        print(
            f"[drpai_runtime] ONNX Runtime EMULATION backend ({onnx_files[0]}, "
            f"input='{self._ort_input}') — NOT the NPU/MERA runtime"
        )
        return True

    def set_input(self, index, data):
        arr = np.ascontiguousarray(data, dtype=np.float32)
        if self._backend == "tvm":
            import tvm

            self._mod.set_input(self._input_name, tvm.nd.array(arr, self._dev))
        else:
            self._feed = arr

    def run(self):
        if self._backend == "tvm":
            self._mod.run()
        else:
            self._outputs = self._sess.run(None, {self._ort_input: self._feed})

    def num_input(self):
        return 1

    def num_output(self):
        if self._backend == "tvm":
            return self._mod.get_num_outputs()
        return len(self._outputs) if self._outputs is not None else 0

    def get_output(self, index):
        # Flat buffer, like the C++ GetOutput; the engine reshapes it.
        if self._backend == "tvm":
            return self._mod.get_output(index).numpy().reshape(-1).astype(np.float32)
        return np.asarray(self._outputs[index], dtype=np.float32).reshape(-1)
