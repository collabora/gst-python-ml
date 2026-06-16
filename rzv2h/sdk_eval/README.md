# Faithful DRP-AI TVM eval (real mera2 / MERA runtime)

This is the most faithful test short of running on hardware: the **real**
`mera2` compile and the **real** MERA/TVM runtime, instead of the ONNX-RT
look-alike in [../emulation](../emulation). It composes with the same
`engine-name=drpai` + `drpai_runtime` shim we use everywhere else.

## Read this first — what's gated, and the aarch64 catch

Two things make this unable to run on a plain x86 box out of the box:

1. **License-gated downloads (Renesas account required).** The stack build needs
   the **DRP-AI Translator i8** and the **RZ/V2H AI SDK** (`RTK0EF0180F06000SJ.zip`).
   There is **no public prebuilt image**; you download these and build Renesas'
   `Dockerfile`. I cannot fetch them for you.
2. **The compile targets aarch64, not x86.** Even `compile_cpu_only_onnx_model.py`
   uses `target = "llvm ... -mtriple=aarch64-linux-gnu"` and the SDK's aarch64
   cross-g++. So `deploy.so` runs on the board's Arm CPU / NPU — to execute it
   off-board you either run on the **board**, under **QEMU-aarch64**, or compile
   with an **x86 `llvm` target** for a pure functional check (see below).

If you don't have the downloads, the ONNX-RT emulation in `../emulation`
already validates all of *our* code (engine preprocess/reshape/decode +
pipeline). What's left to validate here is mera2-compile success and runtime
numerics — both inherently need Renesas assets or hardware.

## Steps

### 1. Build the SDK image (host, needs the two downloads)

```bash
mkdir -p rzv2h/sdk_eval/assets
# put both Renesas downloads in rzv2h/sdk_eval/assets/ :
#   DRP-AI_Translator_i8-*-Linux-x86_64-Install   and   RTK0EF0180F06000SJ.zip
cd rzv2h/sdk_eval && ./build_image.sh
```

`build_image.sh` fetches the repo `Dockerfile`, assembles a clean build context
(Dockerfile + the toolchain `.sh` it unzips from the AI SDK zip + the Translator
installer), and runs `docker build --build-arg PRODUCT=V2H -t drpai-tvm-v2h`.
The Dockerfile (`FROM ubuntu:22.04`) defaults `PRODUCT=V2H` and builds the TVM
fork itself, so the build takes a while.

To fetch just the Dockerfile by hand:
`wget https://raw.githubusercontent.com/renesas-rz/rzv_drp-ai_tvm/main/Dockerfile`

### 2. Compile YOLO11 with the real mera2 (inside the container)

```bash
docker run -it --rm -v "$PWD":/workspace/gst-python-ml drpai-tvm-v2h bash
# inside:
cd /workspace/gst-python-ml
./rzv2h/convert_yolo11_v2h.sh yolo11m 640      # real mera2.from_onnx + mera2.drp.build
# -> yolo11m_drpai_v2h/{deploy.so,deploy.json,deploy.params}   (aarch64)
```

For a **host x86 functional check** instead of the board artifact, compile with a
native target (edit a copy of `tutorials/compile_onnx_model.py` to
`target = "llvm"` and drop the aarch64 cross-compiler), producing an x86
`deploy.so` the MERA/TVM `graph_executor` can run natively.

### 3. Run through the real MERA/TVM runtime

The [../emulation/drpai_runtime.py](../emulation/drpai_runtime.py) shim
auto-selects the **MERA/TVM `graph_executor`** backend as soon as the model dir
has `deploy.so/json/params` and `tvm` is importable (true inside this
container). The engine code is unchanged.

```bash
export GST_PLUGIN_PATH=/workspace/gst-python-ml/plugins:$GST_PLUGIN_PATH
export PYTHONPATH=/workspace/gst-python-ml/rzv2h/emulation:$PYTHONPATH
# (x86 deploy.so) run natively; (aarch64 deploy.so) run under qemu-aarch64
gst-launch-1.0 filesrc location=08fd33_4.mp4 ! decodebin ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! pyml_objectdetector engine-name=drpai model-name=yolo11m_drpai_v2h device=drpai \
        input-format=nchw post-process=anchor_free \
  ! pyml_tracker ! videoconvert ! "video/x-raw,format=RGBA" \
  ! pyml_football_overlay ! videoconvert ! autovideosink
```

The shim prints which backend it picked:
`[drpai_runtime] MERA/TVM graph_executor backend ... — real runtime`.

## On the actual board

Two ways to run the same pipeline on the RZ/V2H:

- **Python graph_executor** — copy the `deploy.so/json/params` + the emulation
  shim; if the board image has the MERA/TVM python runtime, it Just Works (the
  shim's TVM backend), NPU included.
- **C++ pybind binding** — build [../drpai_runtime_pybind.cpp](../drpai_runtime_pybind.cpp)
  per [../README.md](../README.md); the native `drpai_runtime.so` takes
  precedence over this shim on `PYTHONPATH`.

## Verified results (RZ/V2H AI SDK v6.00 + DRP-AI Translator i8 v1.11)

Both paths were run end-to-end driving the `drpai-tvm-v2h` image on an x86 host:

- **x86 MERA/TVM runtime test** — `compile_x86_cpu.py` compiled YOLO11 via the
  MERA-fork TVM (native `llvm`), and `x86_runtime_check.py` ran it through the
  real `graph_executor`: output matched ONNX to **max|Δ| = 6.2e-3**, **22 = 22
  detections** (label `person`). Confirms compile + MERA/TVM runtime + the
  `drpai_runtime` shim + our decoder, no NPU needed.
- **Real INT8 NPU compile** — `../convert_yolo11_v2h.sh` (quantized flow)
  produced the RZ/V2H deploy dir: `[Finish DRP-AI Translator for V2H]`,
  `sub_0000__CPU_DRP_TVM/{deploy.so (65 MB),deploy.json,deploy.params}` +
  `preprocess/` (DRP-AI pre-processing objects). aarch64 — runs on the board.

SDK gotchas the scripts now handle automatically: `run_drp_compiler.sh` ships
non-executable and off-PATH (`chmod +x` + add tutorials to PATH); the quant
script preprocesses calibration as ImageNet-224 instead of 640 (patched). And
V2H **requires** the INT8 quantized flow — the plain FP `compile_onnx_model.py`
drives a legacy translator path the i8 v1.11 layout lacks.
