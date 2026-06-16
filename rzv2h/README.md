# Object detection on Renesas RZ/V2H (DRP-AI NPU)

This runs `pyml_objectdetector` on the **RZ/V2H** DRP-AI NPU, using a YOLO11
model compiled with the **DRP-AI TVM** compiler (powered by EdgeCortix MERA).

It is the decomposed, metadata-passing pipeline used elsewhere in this repo —
detector -> (tracker) -> overlay, but the detector's inference runs on the NPU:

```
... ! pyml_objectdetector engine-name=drpai model-name=<deploy_dir> device=drpai
      input-format=nchw post-process=anchor_free
    ! pyml_tracker ! pyml_overlay ! ...
```
## Prerequisites

- RZ/V2H EVK with the **RZ/V2H AI SDK v6.00** Yocto image (provides the DRP-AI
  driver, `/dev/drpai0`, GStreamer, and Python 3).
- The **DRP-AI TVM** package (`rzv_drp-ai_tvm`) and its SDK Docker, with the
  environment sourced so `TVM_ROOT`, `SDK` (cross SDK), and the DRP-AI
  translator are set. (`PRODUCT=V2H`.)
- `pybind11` headers available to the cross build.

## 1 — Convert the model (in the SDK Docker)

```bash
./convert_yolo11_v2h.sh yolo11m 640
```

This exports YOLO11->ONNX (input node `images`, `1x3x640x640`) and runs the V2H
DRP-AI TVM compiler. See the script for the exact commands.

## 2 — Build the Python binding (in the SDK Docker)

Source the SDK env first (so `TVM_ROOT`/`SDK` are set and CXX is the aarch64
cross compiler), then:

```bash
cd rzv2h
cmake -B build \
  -DCMAKE_TOOLCHAIN_FILE="$TVM_ROOT/apps/toolchain/runtime.cmake" \
  -DPYBIND11_INCLUDE_DIR="$(python3 -m pybind11 --includes | sed 's/-I//;q')" \
  -DPYTHON_INCLUDE_DIR="$SDK/sysroots/aarch64-poky-linux/usr/include/python3.12"
cmake --build build -j
```

Adjust `python3.12` to the AI SDK image's Python version, and point
`PYBIND11_INCLUDE_DIR` at a real pybind11 headers dir if the one-liner doesn't
resolve in the container.

## 3 — Deploy to the board

Copy onto the RZ/V2H (e.g. under `/home/weston`):

- this repo's `plugins/` (the gst-python-ml elements),
- `build/drpai_runtime.so`,
- the compiled `yolo11m_drpai_v2h/` deploy dir,
- a COCO label file if you overlay class names.

```bash
export GST_PLUGIN_PATH=/home/weston/gst-python-ml/plugins:$GST_PLUGIN_PATH
export PYTHONPATH=/home/weston/rzv2h/build:$PYTHONPATH
gst-inspect-1.0 pyml_objectdetector
```

## 4 — Run on the board

File -> annotated file (run as a user that can open `/dev/drpai0`, often root):

```bash
gst-launch-1.0 filesrc location=clip.mp4 ! decodebin ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! pyml_objectdetector engine-name=drpai model-name=yolo11m_drpai_v2h device=drpai \
        input-format=nchw post-process=anchor_free \
  ! pyml_tracker tracker-type=bytetrack \
  ! videoconvert ! "video/x-raw,format=RGBA" ! pyml_overlay \
  ! videoconvert ! autovideosink
```

Live camera (MIPI/USB): swap `filesrc ! decodebin` for the camera source
(`v4l2src` / the EVK's ISP source), keeping the `640x640` caps into the detector.
