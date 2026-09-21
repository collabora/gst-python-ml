# GStreamer Python ML

[![CI](https://github.com/collabora/gst-python-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/collabora/gst-python-ml/actions/workflows/ci.yml)

Pure Python ML elements for upstream GStreamer 1.24 or later.

Video: object detection, zero-shot detection, tracking, pose, depth, zero-shot
classification with CLIP or SigLIP, segmentation with SAM2, OCR, face recognition,
optical flow, super-resolution, action recognition, anomaly detection, captioning,
vision-language models and embeddings. Audio: voice activity detection,
transcription, translation, speech separation, text to speech and audio
classification with CLAP. Text: local and remote LLMs, digests, text to image.
Around them: alerts over webhooks and MQTT, clip recording, metadata sinks and
replay, a Kafka sink, and an MCP server for agents.

Models run through one of the engines: PyTorch, ONNX Runtime, OpenVINO, LiteRT,
TensorFlow, Apache TVM, tinygrad, Apple MLX, ExecuTorch, llama.cpp, Candle, JAX,
MiGraphX, IREE, NCNN and Renesas DRP-AI. CI runs every engine that installs from
PyPI on the CPU and checks its output against PyTorch.

## Table of Contents

- [Install](#install)
  - [Host Install](#host-install)
  - [Docker Install](#docker-install)
- [Post Install](#post-install)
- [Custom Plugins](#custom-plugins)
  - [Directory Structure](#directory-structure)
  - [Example: Custom Object Detector](#example-custom-object-detector)
  - [Environment Setup](#environment-setup)
  - [Available Base Classes](#available-base-classes)
  - [Verify](#verify)
- [Running a pipeline](#running-a-pipeline)
  - [Choosing the backend](#choosing-the-backend)
- [Pipelines](#pipelines)
  - [Classification](#classification)
  - [Torch Compile](#torch-compile)
  - [Object Detection](#object-detection)
  - [Zero-Shot Object Detection](#zero-shot-object-detection)
  - [Pose Estimation](#pose-estimation)
  - [Depth Estimation](#depth-estimation)
  - [Zero-Shot Classification (CLIP / SigLIP)](#zero-shot-classification-clip--siglip)
  - [Voice Activity Detection](#voice-activity-detection)
  - [Transcription](#transcription)
  - [LLM](#llm)
  - [Incident Digest](#incident-digest)
  - [Stable Diffusion](#stable-diffusion)
  - [Caption](#caption)
  - [Kafka Sink](#kafka-sink)
  - [Overlay from a metadata file](#overlay-from-a-metadata-file)
  - [Stream Mux and Demux](#stream-mux-and-demux)
  - [Segment Anything (SAM)](#segment-anything-sam)
  - [OCR](#ocr)
  - [Face Detection & Recognition](#face-detection--recognition)
  - [Optical Flow](#optical-flow)
  - [Super-Resolution](#super-resolution)
  - [Action Recognition](#action-recognition)
  - [Anomaly Detection](#anomaly-detection)
  - [Audio Classification (CLAP)](#audio-classification-clap)
  - [Vision-Language Model (VLM)](#vision-language-model-vlm)
  - [Cascade Gating](#cascade-gating)
  - [Embedding Extractor](#embedding-extractor)
  - [Video Memory](#video-memory)
  - [Multi-Object Tracker](#multi-object-tracker)
  - [ML Alert](#ml-alert)
  - [Alert Recorder](#alert-recorder)
  - [Metadata Sink](#metadata-sink)
  - [Metadata Replay](#metadata-replay)
- [MCP Server](#mcp-server)

## Install

### Host Install

#### Install distribution packages

##### Ubuntu
```
sudo apt update && sudo apt -y upgrade
sudo apt install -y python3-pip  python3-venv \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-base-apps \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gir1.2-gst-plugins-bad-1.0 python3-gst-1.0 gstreamer1.0-python3-plugin-loader \
    libcairo2 libcairo2-dev git
```

##### Fedora

Adjust the Fedora version in the rpmfusion URLs.

```
sudo dnf install https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-42.noarch.rpm https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-42.noarch.rpm
sudo dnf update -y
sudo dnf install akmod-nvidia xorg-x11-drv-nvidia-cuda -y
```

```
sudo dnf upgrade -y
sudo dnf install -y python3-pip \
    python3-devel cairo cairo-devel cairo-gobject-devel pkgconfig git \
    gstreamer1-plugins-base gstreamer1-plugins-base-tools \
    gstreamer1-plugins-good gstreamer1-plugins-bad-free \
    gstreamer1-plugins-bad-free-devel python3-gstreamer1
```



##### Windows

1. Install the runtime and development MSVC x86_64 installers from the
   [GStreamer site](https://gstreamer.freedesktop.org/download/#windows). The default path is `C:\gstreamer\1.0\msvc_x86_64`.

2. Set the environment variables:

```powershell
# Add GStreamer to PATH
[Environment]::SetEnvironmentVariable("PATH", "C:\gstreamer\1.0\msvc_x86_64\bin;" + $env:PATH, "User")

# Point GStreamer at your plugin directory
[Environment]::SetEnvironmentVariable("GST_PLUGIN_PATH", "D:\Workspace\gst-python-ml\plugins", "User")
```

3. Install Python 3.12 or later.

4. Install PyGObject, through conda or the [gstreamer-python](https://pypi.org/project/gstreamer-python/) wheel:

```powershell
pip install gstreamer-python
```

5. For CUDA, install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) matching your driver, then the CUDA PyTorch:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

#### Manage Python packages

##### Python version

GStreamer's Python plugin loader embeds the system interpreter, so the venv must
use the same Python. Ubuntu 24.04 is 3.12, Fedora 42 and Ubuntu 26.04 are 3.14. A
mismatch shows up at run time as `No module named 'torch'`.

##### venv on the system Python

```
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

##### With uv

Point uv at the system Python, not a downloaded one:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python /usr/bin/python3 --system-site-packages
source .venv/bin/activate
uv sync
```

Do not pre-install torch from the PyTorch index here. `uv sync` resolves torch from
`uv.lock`, which on Linux already pulls the CUDA wheels, and replaces whatever was
installed before.

#### Engine extras

Every engine beyond PyTorch is an extra: `onnx`, `onnx-gpu`, `openvino`,
`tensorflow`, `litert`, `tvm`, `tinygrad`, `mlx`, `executorch`, `llamacpp`,
`mlx-cpu`, `jax-cpu`, `jax-gpu`, `jax-tpu`, `iree`, `ncnn`.

```
uv sync --extra onnx
```

- ExecuTorch has no Python 3.14 wheel.
- llama.cpp builds a CPU wheel. For CUDA: `CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python`.
- MLX runs on Apple Silicon. On Linux use the `mlx-cpu` extra.
- Candle has no wheel. Build the bindings: `pip install maturin`, clone
  [candle](https://github.com/huggingface/candle), then `maturin develop -r` in `candle-pyo3`.
- NCNN takes `.param` and `.bin` files. Convert an ONNX model with
  `python -m onnxsim model.onnx model_sim.onnx` (`pip install onnx-simplifier`), then
  `onnx2ncnn model_sim.onnx model.param model.bin`.
- The ONNX engine with `device=npu` needs the [Ryzen AI SDK](https://ryzenai.docs.amd.com/) on AMD Ryzen AI laptops.
- ZenDNN speeds up ONNX Runtime and TensorFlow on AMD EPYC CPUs with
  `ZENDNN_INT8_SUPPORT=1` and `OMP_NUM_THREADS=$(nproc)`, no separate engine.

#### flash-attn

Install the prebuilt wheel matching your Python, torch and CUDA from
[flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases), for example:

```
pip install ./flash_attn-2.8.3+cu128torch2.11-cp314-cp314-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
```

#### MiGraphX (AMD ROCm)

##### Ubuntu

Needs ROCm:
```
sudo apt install migraphx
```

Make the `migraphx` module importable:
```
export PYTHONPATH=/opt/rocm/lib:$PYTHONPATH
```

##### Fedora

Fedora ships ROCm but not MiGraphX, so build it from source against the distro packages.
Tested on Fedora 43 with ROCm 6.4 and a Rembrandt APU (gfx1035).

```
sudo dnf install -y rocminfo rocm-hip-devel rocm-cmake rocblas-devel miopen-devel \
    half-devel protobuf-devel msgpack-devel json-devel sqlite-devel \
    python3-devel python3-pybind11 cmake ninja-build
rocminfo | grep -o -m1 'gfx[0-9a-f]*'
```

Build with the gfx name printed above as `GPU_TARGETS`. The patch adds Python 3.13 and 3.14
to MiGraphX's search list and fixes the build with rocMLIR disabled (Fedora does not package it).

```
cd $HOME/src
git clone --branch rocm-6.4.4 --depth 1 https://github.com/ROCm/AMDMIGraphX.git
cd AMDMIGraphX
git apply $HOME/src/gst-python-ml/extern/migraphx/rocm-6.4.4-fedora.patch
CXX=/usr/lib64/rocm/llvm/bin/clang++ cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=gfx1035 \
    -DMIGRAPHX_ENABLE_MLIR=OFF -DMIGRAPHX_USE_COMPOSABLEKERNEL=OFF -DMIGRAPHX_USE_HIPBLASLT=OFF \
    -DMIGRAPHX_ENABLE_PYTHON=ON -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=$PWD/install
ninja -C build install
```

`hipcc` compiles every file as HIP and the protobuf sources fail to link, hence ROCm's
`clang++`. Make the `migraphx` module importable:
```
export PYTHONPATH=$HOME/src/AMDMIGraphX/install/lib:$PYTHONPATH
```

The first GPU compile of a model takes a couple of minutes. The `device=cpu` reference target
runs but is very slow on detection models.

#### Clone repo

```
cd $HOME/src
git clone https://github.com/collabora/gst-python-ml.git
```

#### Plugin path

```
echo 'export GST_PLUGIN_PATH=$HOME/src/gst-python-ml/plugins:$GST_PLUGIN_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Docker Install

The Dockerfiles mount the checkout from `$HOME/src/gst-python-ml`.

#### Enable Docker GPU Support on Host

Skip this on CPU.


##### Ubuntu
```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

##### Fedora

```
sudo dnf install docker
sudo usermod -aG docker $USER
# Then either log out/in completely, or:
newgrp docker
```


```
# 1. Add NVIDIA Container Toolkit repository
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
  sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

# 2. Remove Fedora's conflicting partial package (if present)
sudo dnf remove -y golang-github-nvidia-container-toolkit 2>/dev/null || true

# 3. Install the full NVIDIA Container Toolkit
sudo dnf install -y nvidia-container-toolkit

# 4. Configure Docker to use the NVIDIA runtime as default
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "runtimes": {
    "nvidia": {
      "path": "/usr/bin/nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-runtime": "nvidia"
}
EOF

# 5. Fix Fedora's broken dockerd ExecStart (required!)
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo tee /etc/systemd/system/docker.service.d/override.conf >/dev/null <<EOF
[Service]
ExecStart=
ExecStart=/usr/bin/dockerd -H fd:// --containerd=/run/containerd/containerd.sock
EOF

# 6. Reload and restart Docker
sudo systemctl daemon-reload
sudo systemctl restart docker

# 7. Verify it works
docker info --format '{{.DefaultRuntime}}'   # → should print: nvidia
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```


#### Build Container

`docker build -f ./Dockerfile_ubuntu24 -t ubuntu24:latest .`

`docker build -f ./Dockerfile_ubuntu26 -t ubuntu26:latest .`

`docker build -f ./Dockerfile_fedora42 -t fedora42:latest .`


#### Run Docker Container

Drop `--gpus all` on CPU.

`docker run -v ~/src/gst-python-ml/:/root/gst-python-ml -it --rm --gpus all --name ubuntu24 ubuntu24:latest /bin/bash`

or

`docker run -v ~/src/gst-python-ml/:/root/gst-python-ml -it --rm --gpus all --name ubuntu26 ubuntu26:latest /bin/bash`

or

`docker run -v ~/src/gst-python-ml/:/root/gst-python-ml -it --rm --gpus all --name fedora42 fedora42:latest /bin/bash`

Then set up the venv in the container shell as above.


## Post Install

Run `gst-inspect-1.0 python` to list pyml elements.

## Custom Plugins

Your own elements can inherit the base classes below from a separate directory.

### Directory Structure

```
my_plugins/
  python/
    my_detector.py
    my_classifier.py
```

### Example: Custom Object Detector

```python
CAN_REGISTER_ELEMENT = True
try:
    import gi
    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    gi.require_version("GObject", "2.0")
    from gi.repository import GObject, Gst, GstBase
    from base_objectdetector import BaseObjectDetector
except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    print(f"my_detector not available: {e}")

if CAN_REGISTER_ELEMENT:
    class MyDetector(BaseObjectDetector):
        __gstmetadata__ = (
            "My Custom Detector",
            "Video/Filter",
            "A custom object detector",
            "Your Name",
        )

    GObject.type_register(MyDetector)
    __gstelementfactory__ = ("my_detector", Gst.Rank.NONE, MyDetector)
```

The plugin loader imports every module on the plugin path at startup, whether or
not the pipeline uses it, so import numpy, torch and the like inside methods.


### Environment Setup

Set both `GST_PLUGIN_PATH` (so GStreamer discovers your `.py` files) and `PYTHONPATH`
(so Python can import your modules):

```bash
export GST_PLUGIN_PATH=$HOME/src/gst-python-ml/plugins:$HOME/my_plugins:$GST_PLUGIN_PATH
export PYTHONPATH=$HOME/my_plugins/python:$PYTHONPATH
```

The loader adds only the first `python/` directory it finds to `sys.path`, so list
this checkout first for the base classes and put your own directory on `PYTHONPATH`.

### Available Base Classes

| Base Class | Module | Description |
|---|---|---|
| `BaseTransform` | `base_transform` | Base for all video transform elements |
| `BaseObjectDetector` | `base_objectdetector` | Object detection with bounding boxes |
| `BaseClassifier` | `base_classifier` | Image classification |
| `BaseCaption` | `base_caption` | Video/image captioning |
| `BaseLLM` | `base_llm` | Large language models |
| `BaseTranscribe` | `base_transcribe` | Speech-to-text transcription |
| `BaseTranslate` | `base_translate` | Text translation |
| `BaseTTS` | `base_tts` | Text-to-speech synthesis |
| `BaseSeparate` | `base_separate` | Audio source separation |

### Verify

```bash
gst-inspect-1.0 my_detector
```

## Running a pipeline

`pyml-launch` runs every pipeline in this README. Run it from the checkout with
any Python: it re-runs itself under `.venv` if it finds one, so the elements see
torch and the rest. Installing the package also puts a `pyml-launch` on `PATH`.

```bash
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale \
  ! video/x-raw,width=640,height=480 \
  ! pyml_yolo model-name=yolo11m device=cuda:0 track=True \
  ! pyml_overlay ! videoconvert ! autovideosink
```

### Choosing the backend

The ML elements run under GStreamer or under
[glass2glass](https://gitlab.collabora.com/glass2glass/glass2glass), selected by
`PYML_BACKEND`. It defaults to `gst`, which is why the examples leave it off.
Prefix any of them with `PYML_BACKEND=g2g` to run the same line under
glass2glass instead.

Under `gst` it runs GStreamer with `GST_PLUGIN_PATH` pointing at this checkout.
Under `g2g` it runs `g2g-launch-py`, rewriting the three things g2g spells
differently: a `pyml_*` element becomes `pyelement` plus the module and class to
host, an element g2g implements itself becomes that one, and a raw-video caps
filter with no format gains `format=RGBA`. An element it cannot map is an error,
not a silent pass-through.

Seven elements are native on g2g rather than hosted: `pyml_overlay` becomes
`analyticsoverlay`, `pyml_alert` becomes `analyticsalert`, `pyml_digest` becomes
`textdigest`, and `pyml_metasink`, `pyml_metareplay`, `pyml_alertrecorder` and
`pyml_embeddingsink` drop the prefix. Their `location`, `rules`, `cooldown`,
`draw-alert`, `webhook-url`, `encoder`, `seconds-before`, `seconds-after`,
`window-seconds`, `source-id` and `model-name` carry over under the same names,
and the file formats match, so `search_video` reads an index either backend
wrote. g2g runs no MQTT, so `pyml_alert`'s `mqtt-broker` and `mqtt-topic` are
refused rather than dropped.

`analyticsoverlay` has its own properties (`show-label`, `show-track`,
`show-score`, `show-trail`, `trail-length`, `thickness`, `mask-alpha`), so only
`pyml_overlay`'s `tracking` carries over, as `show-track`.

Build `g2g-launch-py` from a glass2glass checkout and put it on `PATH`, or point
`G2G_LAUNCH` at it:

```bash
PYO3_PYTHON=$(which python) cargo build --release -p g2g-python --features ml \
  --bin g2g-launch-py
```

## Pipelines

One or two lines per element. Paths are relative to the checkout.

### Classification

```
python pyml-launch.py  filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! pyml_classifier model-name=resnet18 device=cuda !  videoconvert !  autovideosink
```


### Torch Compile

Every PyTorch element takes `compile=True` to run the model through `torch.compile`,
which trades a slow first frame for higher steady throughput.

#### Classification with torch.compile

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale \
  ! video/x-raw,width=640,height=480 \
  ! pyml_classifier model-name=resnet18 device=cuda compile=True \
  ! videoconvert ! autovideosink
```

#### Object detection with torch.compile

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale \
  ! video/x-raw,width=640,height=480 \
  ! pyml_objectdetector model-name=fasterrcnn_resnet50_fpn device=cuda compile=True \
  ! videoconvert ! pyml_overlay ! videoconvert ! autovideosink
```


### Object Detection

#### TorchVision

`pyml_objectdetector` takes any torchvision detection model name, such as
`fasterrcnn_resnet50_fpn` or `ssdlite320_mobilenet_v3_large`.

##### fasterrcnn

`python pyml-launch.py  filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! pyml_objectdetector model-name=fasterrcnn_resnet50_fpn device=cuda batch-size=4 ! videoconvert ! pyml_overlay ! videoconvert ! autovideosink`

##### fasterrcnn/kafka

From the host:

```
python pyml-launch.py  filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! pyml_objectdetector model-name=fasterrcnn_resnet50_fpn device=cuda batch-size=4 ! pyml_kafkasink schema-file=data/pyml_object_detector.json broker=localhost:29092 topic=test-kafkasink-topic
```

From a container:

```
python pyml-launch.py  filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! pyml_objectdetector model-name=fasterrcnn_resnet50_fpn device=cuda batch-size=4 ! pyml_kafkasink schema-file=data/pyml_object_detector.json broker=kafka:9092 topic=test-kafkasink-topic
```


#### maskrcnn

```
python pyml-launch.py   filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! pyml_maskrcnn device=cuda batch-size=4 model-name=maskrcnn_resnet50_fpn ! videoconvert ! pyml_overlay ! videoconvert ! autovideosink
```

#### yolo with tracking

```
python pyml-launch.py   filesrc location=data/soccer_tracking.mp4 ! decodebin !  videoconvertscale ! video/x-raw,width=640,height=480 ! pyml_yolo model-name=yolo11m device=cuda:0 track=True ! pyml_overlay  ! videoconvert ! autovideosink
```

```
python pyml-launch.py   filesrc location=data/soccer_tracking.mp4 ! decodebin ! videoconvertscale ! video/x-raw,width=640,height=480,format=RGB ! pyml_streammux name=mux   filesrc location=data/soccer_tracking.mp4 ! decodebin ! videoconvertscale ! video/x-raw,width=640,height=480,format=RGB ! mux.   mux. ! pyml_yolo model-name=yolo11m device=cuda:0 track=True ! pyml_streamdemux name=demux   demux. ! queue ! videoconvert ! pyml_overlay ! videoconvert ! autovideosink sync=false   demux. ! queue ! videoconvert ! pyml_overlay ! videoconvert !  autovideosink sync=false

```

```
python pyml-launch.py filesrc location=data/soccer_tracking.mp4 ! decodebin ! videoconvertscale ! video/x-raw,width=640,height=480 ! demo_soccer model-name=yolo11m device=cuda:0 ! pyml_overlay ! videoconvert ! autovideosink
```


#### ONNX Engine

Export a YOLO11 model with ultralytics:

```
yolo export model=yolo11m.pt format=onnx
```

##### YOLO11m ONNX object detection with overlay

YOLO takes channels first, hence `input-format=nchw`. `post-process=anchor_free`
decodes the raw `[B, 4+nc, anchors]` output into boxes with NMS.

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! pyml_objectdetector engine-name=onnx model-name=yolo11m.onnx device=cpu \
              input-format=nchw post-process=anchor_free \
  ! videoconvert ! "video/x-raw,format=RGBA" \
  ! pyml_overlay ! videoconvert ! autovideosink
```

##### Generic ONNX passthrough (logs raw inference output)

`pyml_inference` runs any model and logs the raw output:

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! pyml_inference engine-name=onnx model-name=yolo11m.onnx device=cpu \
  ! fakesink
```

It takes every `engine-name` below.

#### OpenVINO Engine


```
yolo export model=yolo11m.pt format=openvino
```


##### YOLO11m OpenVINO object detection with overlay

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! pyml_objectdetector engine-name=openvino \
              model-name=yolo11m_openvino_model/yolo11m.xml device=cpu \
              input-format=nchw post-process=anchor_free \
  ! videoconvert ! "video/x-raw,format=RGBA" \
  ! pyml_overlay ! videoconvert ! autovideosink
```

`device=GPU` targets an Intel GPU. OpenVINO device names are uppercase.

#### LiteRT (TFLite) Engine


```
yolo export model=yolo11m.pt format=tflite
```


##### YOLO11m TFLite object detection with overlay

The engine reads the input layout from the model and scales the normalized boxes an
ultralytics export returns back to pixels.

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! pyml_objectdetector engine-name=tflite \
              model-name=yolo11m_saved_model/yolo11m_float32.tflite device=cpu \
              post-process=anchor_free \
  ! videoconvert ! "video/x-raw,format=RGBA" \
  ! pyml_overlay ! videoconvert ! autovideosink
```

#### TensorFlow Engine


```
yolo export model=yolo11m.pt format=saved_model
```

##### YOLO11m TensorFlow object detection with overlay

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! pyml_objectdetector engine-name=tensorflow \
              model-name=yolo11m_saved_model device=cuda \
              post-process=anchor_free \
  ! videoconvert ! "video/x-raw,format=RGBA" \
  ! pyml_overlay ! videoconvert ! autovideosink
```

#### tinygrad Engine

tinygrad runs the torchvision resnet family (resnet, resnext, wide_resnet) from
torchvision weights.

##### ResNet18 classification with tinygrad on GPU

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=224,height=224" \
  ! pyml_classifier model-name=resnet18 device=cuda engine-name=tinygrad \
  ! fakesink
```

##### tinygrad on CPU

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=224,height=224" \
  ! pyml_classifier model-name=resnet18 device=cpu engine-name=tinygrad \
  ! fakesink
```

#### TVM Engine

TVM takes a compiled `.so` or `.tar`, or a torchvision model name, which it exports
with torch.export and compiles through relax at load time.

##### TorchVision model compiled with TVM

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=224,height=224" \
  ! pyml_classifier model-name=resnet18 device=cuda engine-name=tvm \
  ! fakesink
```

##### Pre-compiled TVM model (.so)

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! pyml_inference engine-name=tvm model-name=compiled_model.so device=cuda \
  ! fakesink
```

#### Apple MLX Engine

MLX takes the torchvision resnet family, SafeTensors or `.npz` weights, and mlx-lm models.

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=224,height=224" \
  ! pyml_classifier model-name=resnet18 device=gpu engine-name=mlx \
  ! fakesink
```

#### ExecuTorch Engine

ExecuTorch runs `.pte` files, for example `yolo export model=yolo11n.pt format=executorch`.

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=224,height=224" \
  ! pyml_inference engine-name=executorch model-name=model.pte device=cpu \
  ! fakesink
```

#### llama.cpp Engine

llama.cpp runs `.gguf` files.

```
python pyml-launch.py filesrc location=data/prompt_for_llm.txt \
  ! pyml_llm engine-name=llamacpp model-name=model.gguf device=cpu \
  ! fakesink
```

#### Candle Engine

Candle takes SafeTensors files.

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=224,height=224" \
  ! pyml_inference engine-name=candle model-name=model.safetensors device=cpu \
  ! fakesink
```

#### JAX/Flax Engine

JAX takes the torchvision resnet family and Flax checkpoints.

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=224,height=224" \
  ! pyml_classifier model-name=resnet18 device=cpu engine-name=jax \
  ! fakesink
```

#### MiGraphX Engine

MiGraphX takes ONNX files and needs the ROCm install above.

##### YOLO11m MiGraphX object detection with overlay

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! pyml_objectdetector engine-name=migraphx model-name=yolo11m.onnx device=gpu \
              input-format=nchw post-process=anchor_free \
  ! videoconvert ! "video/x-raw,format=RGBA" \
  ! pyml_overlay ! videoconvert ! autovideosink
```

##### MiGraphX on CPU (reference target)

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! pyml_inference engine-name=migraphx model-name=yolo11m.onnx device=cpu \
  ! fakesink
```

#### IREE Engine

IREE takes a compiled `.vmfb`, or an `.onnx` it compiles at load time, for `hip`,
`vulkan`, `cuda` or `cpu`.

##### IREE on AMD GPU (ROCm/HIP)

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! pyml_inference engine-name=iree model-name=yolo11m.onnx device=hip \
  ! fakesink
```

##### IREE on Vulkan (any GPU vendor)

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! pyml_inference engine-name=iree model-name=yolo11m.onnx device=vulkan \
  ! fakesink
```

##### IREE with pre-compiled module

```
# Pre-compile: iree-compile model.mlir --iree-hal-target-device=hip -o model.vmfb
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! pyml_inference engine-name=iree model-name=model.vmfb device=hip \
  ! fakesink
```

#### NCNN Engine (Vulkan)

NCNN takes a `.param` with its `.bin` alongside and runs on any Vulkan GPU or the CPU.

##### NCNN on Vulkan GPU

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! pyml_inference engine-name=ncnn model-name=yolo11m.param device=vulkan \
  ! fakesink
```

##### NCNN on CPU

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! pyml_inference engine-name=ncnn model-name=yolo11m.param device=cpu \
  ! fakesink
```

#### ONNX Runtime on AMD GPUs (ROCm)

`device=rocm` picks the MIGraphX execution provider, falling back to ROCm's.

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! pyml_objectdetector engine-name=onnx model-name=yolo11m.onnx device=rocm \
              input-format=nchw post-process=anchor_free \
  ! videoconvert ! "video/x-raw,format=RGBA" \
  ! pyml_overlay ! videoconvert ! autovideosink
```

#### ONNX Runtime on AMD Ryzen AI NPU

`device=npu` needs the Ryzen AI SDK.

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! pyml_objectdetector engine-name=onnx model-name=yolo11m.onnx device=npu \
              input-format=nchw post-process=anchor_free \
  ! videoconvert ! "video/x-raw,format=RGBA" \
  ! pyml_overlay ! videoconvert ! autovideosink
```

#### PyTorch on AMD GPUs (ROCm)

A ROCm torch answers to `device=cuda`:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
```

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! pyml_objectdetector model-name=fasterrcnn_resnet50_fpn device=cuda \
  ! videoconvert ! pyml_overlay ! videoconvert ! autovideosink
```

With `torch.compile`, which uses Triton on AMD:

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! pyml_objectdetector model-name=fasterrcnn_resnet50_fpn device=cuda compile=True \
  ! videoconvert ! pyml_overlay ! videoconvert ! autovideosink
```

### Zero-Shot Object Detection

`pyml_zeroshotdetector` detects the classes named in its `labels` property, so
`pyml_alert`, `pyml_tracker`, `pyml_overlay` and `pyml_metasink` work on a class
no detector was trained for. Detections carry the label text, as
`stream_0_handbag`.

Supported models:
```
google/owlv2-base-patch16-ensemble  (default)
IDEA-Research/grounding-dino-tiny   (faster, smaller)
```

#### Alert on a class no detector was trained for

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! pyml_zeroshotdetector device=cuda labels="person, handbag" confidence=0.2 \
  ! pyml_alert rules='{"class":"handbag"}' \
  ! pyml_overlay ! videoconvert ! autovideosink sync=false
```

### Pose Estimation

`pyml_yolo_pose` takes any YOLO pose model:
```
yolo11n-pose  (fastest)
yolo11s-pose
yolo11m-pose  (best accuracy)
```

#### YOLO pose with skeleton visualization (rendered on frame)

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue \
    ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
    ! pyml_yolo_pose model-name=yolo11n-pose device=cuda \
    ! videoconvert ! autovideosink sync=false
```

#### YOLO pose with bounding box overlay (metadata only, no in-element rendering)

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue \
    ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
    ! pyml_yolo_pose model-name=yolo11n-pose device=cuda visualize=false \
    ! videoconvert ! pyml_overlay ! videoconvert ! autovideosink sync=false
```

### Depth Estimation

`pyml_depth` takes the Depth Anything V2 models:
```
depth-anything/Depth-Anything-V2-Small-hf  (fastest, ~100 MB)
depth-anything/Depth-Anything-V2-Base-hf
depth-anything/Depth-Anything-V2-Large-hf  (most accurate)
```

Colormaps: `inferno` (default), `jet`, `viridis`, `plasma`, `magma`.

#### DepthAnything V2 with inferno colormap

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue \
    ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
    ! pyml_depth model-name=depth-anything/Depth-Anything-V2-Small-hf device=cuda \
    ! videoconvert ! autovideosink sync=false
```

#### DepthAnything V2 with jet colormap

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue \
    ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
    ! pyml_depth model-name=depth-anything/Depth-Anything-V2-Small-hf device=cuda colormap=jet \
    ! videoconvert ! autovideosink sync=false
```

#### Depth with reduced compute via frame-stride

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue \
    ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
    ! pyml_depth model-name=depth-anything/Depth-Anything-V2-Small-hf device=cuda frame-stride=2 \
    ! videoconvert ! autovideosink sync=false
```

#### Depth with original video side-by-side (tee)

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue \
    ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
    ! tee name=t \
    t. ! queue ! pyml_depth model-name=depth-anything/Depth-Anything-V2-Small-hf device=cuda ! videoconvert ! autovideosink sync=false \
    t. ! queue ! videoconvert ! autovideosink sync=false
```

### Zero-Shot Classification (CLIP / SigLIP)

`pyml_clip` classifies each frame against the text labels you pass.

Supported models:
```
openai/clip-vit-base-patch32       (default, ~600 MB)
openai/clip-vit-large-patch14      (more accurate, ~1.7 GB)
google/siglip-base-patch16-224     (SigLIP, better zero-shot accuracy)
google/siglip-large-patch16-384    (SigLIP large)
```

#### CLIP with custom labels

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue \
    ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
    ! pyml_clip model-name=openai/clip-vit-base-patch32 device=cuda \
              labels="person, bicycle, car, dog, cat" top-k=3 \
    ! videoconvert ! pyml_overlay ! videoconvert ! autovideosink sync=false
```

#### SigLIP (better zero-shot accuracy than CLIP)

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue \
    ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
    ! pyml_clip model-name=google/siglip-base-patch16-224 device=cuda \
              labels="people walking, empty street, crowd, indoor scene" top-k=1 \
    ! videoconvert ! pyml_overlay ! videoconvert ! autovideosink sync=false
```

#### CLIP with threshold (only report labels above 20% confidence)

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue \
    ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
    ! pyml_clip model-name=openai/clip-vit-base-patch32 device=cuda \
              labels="person, bicycle, car, dog, cat" threshold=0.2 \
    ! videoconvert ! pyml_overlay ! videoconvert ! autovideosink sync=false
```

### Voice Activity Detection

#### Standalone VAD with metadata (pass-through, speech probability attached to buffers)

```
python pyml-launch.py pulsesrc ! audio/x-raw,format=S16LE,rate=16000,channels=1 ! pyml_vad threshold=0.7 ! fakesink
```

`pyml_whispertranscribe` has its own VAD to split speech into clips, so `pyml_vad` is not needed in front of it.

### Transcription

Transcripts are logged at GStreamer info level. Run with `GST_DEBUG=python:4` to see them.

faster-whisper's CTranslate2 wheel loads CUDA 12 cuBLAS. With a CUDA 13 torch install, add it to the venv and the loader path. On Linux:

```
pip install nvidia-cublas-cu12
export LD_LIBRARY_PATH=$(python -c "import nvidia.cublas, os; print(os.path.join(nvidia.cublas.__path__[0], 'lib'))"):$LD_LIBRARY_PATH
```

On Windows add the same package's `bin` directory to `PATH` instead.

#### live microphone (Linux)

`pulsesrc` works on PulseAudio (Ubuntu) and on PipeWire's Pulse server (Fedora). `pipewiresrc` does not work with GStreamer 1.26 (PipeWire 1.4): it stalls after a few buffers even into `fakesink`, because its buffer timestamps and the clock it provides disagree and GstBaseSrc's clock wait hangs. `do-timestamp=true` helps but is not consistent.

```
python pyml-launch.py pulsesrc ! audioconvert ! audioresample ! audio/x-raw,format=S16LE,rate=16000,channels=1 ! pyml_whispertranscribe device=cuda language=en ! fakesink
```

```
python pyml-launch.py pipewiresrc do-timestamp=true ! audioconvert ! audioresample ! audio/x-raw,format=S16LE,rate=16000,channels=1 ! pyml_whispertranscribe device=cuda language=en ! fakesink
```

If nothing is transcribed, check which port the source captures from with `pactl list sources | grep "Active Port"`. Laptop combo jacks often report a plugged-in headset mic as not available and keep the internal mic. Force it with `pactl set-source-port <source> analog-input-mic`. On PipeWire, WirePlumber then drops that source as the default, so name it: `pulsesrc device=<source>` or `pipewiresrc target-object=<source>`. On Windows and macOS use `autoaudiosrc` and pick the input device in the OS sound settings.

#### transcription with initial prompt set

```
python pyml-launch.py filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! pyml_whispertranscribe device=cuda language=ko initial_prompt = "Air Traffic Control은, radar systems를,  weather conditions에, flight paths를, communication은, unexpected weather conditions가, continuous training을, dedication과, professionalism" ! fakesink
```

#### translation to English

```
python pyml-launch.py filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! pyml_whispertranscribe device=cuda language=ko translate=yes ! fakesink
```

#### demucs audio separation

```
python pyml-launch.py filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! audioresample ! pyml_demucs device=cuda ! wavenc ! filesink location=separated_vocals.wav
```

#### sepformer audio separation

`stem` picks the output, `vocals` by default.

```
python pyml-launch.py filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! audioresample ! pyml_sepformer device=cuda ! wavenc ! filesink location=separated_vocals.wav
```


#### whisperspeechtts

```
python pyml-launch.py filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! pyml_whispertranscribe device=cuda language=ko translate=yes ! pyml_whisperspeechtts device=cuda ! audioconvert ! wavenc ! filesink location=output_audio.wav
```

#### mariantranslate

```
python pyml-launch.py filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! pyml_whispertranscribe device=cuda language=ko translate=yes ! pyml_mariantranslate device=cuda src=en target=fr ! fakesink
```

`src` and `target` take any pair with a [Helsinki-NLP opus-mt model](https://huggingface.co/models?search=Helsinki).


#### whisperlive

`python pyml-launch.py filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! pyml_whisperlive device=cuda language=ko translate=yes llm-model-name="microsoft/phi-2" ! audioconvert ! wavenc ! filesink location=output_audio.wav`

### LLM

Gated models need `hf auth login` first.

`python pyml-launch.py filesrc location=data/prompt_for_llm.txt !  pyml_llm device=cuda model-name="microsoft/phi-2" ! fakesink`

#### Remote LLM

`pyml_llm_remote` posts text to an LLM server. The OpenAI-compatible
`/v1/chat/completions` path works against Ollama, llama.cpp and vLLM. Without
`url=` it uses Ollama's native `/api/generate`, and it picks the request format
from the path.

##### Basic call

```
python pyml-launch.py filesrc location=data/prompt_for_llm.txt \
  ! "text/x-raw,format=utf8" \
  ! pyml_llm_remote url=http://localhost:11434/v1/chat/completions \
    model-name=llama3 \
  ! fakesink
```

##### With a system prompt and a custom model

```
python pyml-launch.py filesrc location=data/prompt_for_llm.txt \
  ! "text/x-raw,format=utf8" \
  ! pyml_llm_remote url=http://localhost:11434/v1/chat/completions \
    model-name=qwen3:8b \
    system-prompt="You are a helpful assistant. Answer concisely." \
    temperature=0.5 \
  ! fakesink
```

### Incident Digest

`pyml_digest` collects the text buffers of a time window into one text buffer,
so `pyml_llm` summarises half a minute of captions rather than a single line.
The window closes when a buffer's timestamp passes it, and the open window is
flushed at end of stream.

```
python pyml-launch.py filesrc location=data/soccer_single_camera.mp4 ! decodebin ! videoconvertscale ! video/x-raw,width=640,height=480 ! tee name=t t. ! queue ! textoverlay name=overlay wait-text=false ! videoconvert ! autovideosink t. ! queue leaky=2 max-size-buffers=1 ! videoconvertscale ! video/x-raw,width=240,height=180 ! pyml_caption_qwen device=cuda:0 prompt="In one sentence, describe what you see?" model-name="Qwen/Qwen2.5-VL-3B-Instruct-AWQ" name=cap cap.src ! fakesink async=0 sync=0 cap.text_src ! queue ! pyml_digest window-seconds=30 ! pyml_llm model-name="Qwen/Qwen3-0.6B" device=cuda system-prompt="You receive every caption of the last thirty seconds. Write one paragraph describing what happened, and NEVER mention the specific times." ! queue ! overlay.text_sink
```

### Stable Diffusion

`python pyml-launch.py filesrc location=data/prompt_for_stable_diffusion.txt ! pyml_stablediffusion device=cuda ! pngenc ! filesink location=output_image.png`

### Caption

`pyml_caption_qwen` captions frames with Qwen2.5-VL, `pyml_caption_phi` with
Phi-3.5-vision. Here `coalescehistory` hands the last ten captions to an LLM for a
running summary:

```
python pyml-launch.py filesrc location=data/soccer_single_camera.mp4 ! decodebin ! videoconvertscale ! video/x-raw,width=640,height=480 ! tee name=t t. ! queue ! textoverlay name=overlay wait-text=false ! videoconvert ! autovideosink t. ! queue leaky=2 max-size-buffers=1 ! videoconvertscale ! video/x-raw,width=240,height=180 ! pyml_caption_qwen device=cuda:0 prompt="In one sentence, describe what you see?" model-name="Qwen/Qwen2.5-VL-3B-Instruct-AWQ" name=cap cap.src ! fakesink async=0 sync=0 cap.text_src ! queue ! coalescehistory history-length=10 ! pyml_llm model-name="Qwen/Qwen3-0.6B" device=cuda system-prompt="You receive the history of what happened in recent times, summarize it nicely with excitement but NEVER mention the specific times. Focus on the most recent events." ! queue ! overlay.text_sink
```

### Kafka Sink

`pyml_kafkasink` needs a broker. Create a docker network and add `--network kafka-network`
to the `docker run` line above so a containerised pipeline reaches it:

```
docker network create kafka-network
```

#### Set up kafka and zookeeper

From the host the broker is `localhost:29092` instead of `kafka:9092`.

```
docker stop kafka zookeeper
docker rm kafka zookeeper
docker run -d --name zookeeper --network kafka-network -e ZOOKEEPER_CLIENT_PORT=2181 confluentinc/cp-zookeeper:latest
docker run -d --name kafka --network kafka-network \
  -e KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181 \
  -e KAFKA_ADVERTISED_LISTENERS=INSIDE://kafka:9092,OUTSIDE://localhost:29092 \
  -e KAFKA_LISTENER_SECURITY_PROTOCOL_MAP=INSIDE:PLAINTEXT,OUTSIDE:PLAINTEXT \
  -e KAFKA_LISTENERS=INSIDE://0.0.0.0:9092,OUTSIDE://0.0.0.0:29092 \
  -e KAFKA_INTER_BROKER_LISTENER_NAME=INSIDE \
  -e KAFKA_BROKER_ID=1 \
  -e KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1 \
  -p 9092:9092 \
  -p 29092:29092 \
  confluentinc/cp-kafka:latest
```

#### Topics

```
docker exec kafka kafka-topics --create --topic test-kafkasink-topic --bootstrap-server kafka:9092 --partitions 1 --replication-factor 1
docker exec -it kafka kafka-topics --list --bootstrap-server kafka:9092
docker exec -it kafka kafka-topics --delete --topic test-topic --bootstrap-server kafka:9092
docker exec -it kafka kafka-console-consumer --bootstrap-server kafka:9092 --topic test-kafkasink-topic --from-beginning
```

### Overlay from a metadata file

`pyml_overlay` and `pyml_overlay_counter` draw the detections in `meta-path` with no model in the pipeline.

`python pyml-launch.py videotestsrc ! video/x-raw,width=1280,height=720 ! pyml_overlay meta-path=data/sample_metadata.json tracking=true ! videoconvert ! autovideosink`

`python pyml-launch.py videotestsrc ! video/x-raw,width=1280,height=720 ! pyml_overlay_counter meta-path=data/sample_metadata.json tracking=true ! videoconvert ! autovideosink`


### Stream Mux and Demux

```
 python pyml-launch.py   videotestsrc pattern=ball ! video/x-raw, width=320, height=240 ! queue ! pyml_streammux name=mux   videotestsrc pattern=smpte ! video/x-raw, width=320, height=240 ! queue ! mux.sink_1   videotestsrc pattern=smpte ! video/x-raw, width=320, height=240 ! queue ! mux.sink_2   mux.src ! queue ! pyml_streamdemux name=demux   demux.src_0 ! queue ! glimagesink  demux.src_1 ! queue ! glimagesink   demux.src_2 ! queue  ! glimagesink
```

### Segment Anything (SAM)

`pyml_sam` runs SAM2 with point, box or automatic prompts.

#### Auto-mask segmentation (segment everything)

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! pyml_sam model-name=facebook/sam2-hiera-small device=cuda mode=auto \
  ! videoconvert ! autovideosink sync=false
```

#### Point-prompt segmentation (segment object at center)

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! pyml_sam model-name=facebook/sam2-hiera-small device=cuda \
            mode=points max-masks=10 \
  ! videoconvert ! autovideosink sync=false
```

### OCR

`pyml_ocr` recognizes text with TrOCR and appends it as a `GST-OCR:` chunk.

#### TrOCR recognition

```
python pyml-launch.py filesrc location=data/document.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! pyml_ocr model-name=microsoft/trocr-base-printed device=cuda \
  ! videoconvert ! pyml_overlay ! videoconvert ! autovideosink sync=false
```

### Face Detection & Recognition

`pyml_face` detects faces with RetinaFace and names them from ArcFace embeddings of a gallery.

#### Face detection only

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! pyml_face device=cuda \
  ! videoconvert ! pyml_overlay ! videoconvert ! autovideosink sync=false
```

#### Face detection + recognition with gallery

`gallery-path` holds one image per person, named after them. Without it faces are detected but not named.

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! pyml_face device=cuda gallery-path=data/face_gallery/ threshold=0.6 \
  ! videoconvert ! pyml_overlay ! videoconvert ! autovideosink sync=false
```

### Optical Flow

`pyml_optical_flow` runs RAFT between consecutive frames.

#### RAFT optical flow with color visualization

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! pyml_optical_flow model-name=raft-small device=cuda visualize=true \
  ! videoconvert ! autovideosink sync=false
```

### Super-Resolution

`pyml_superres` upscales with Real-ESRGAN.

#### 2x upscale

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=320,height=240" \
  ! pyml_superres device=cuda scale-factor=2 \
  ! videoconvert ! autovideosink sync=false
```

#### 4x upscale with tile processing

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=320,height=240" \
  ! pyml_superres device=cuda scale-factor=4 \
  ! videoconvert ! autovideosink sync=false
```

### Action Recognition

`pyml_action` classifies the action in a sliding window of frames.

#### SlowFast action recognition

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! pyml_action model-name=slowfast_r50 device=cuda num-frames=32 \
  ! videoconvert ! pyml_overlay ! videoconvert ! autovideosink sync=false
```

### Anomaly Detection

`pyml_anomaly` scores frames against PatchCore features of normal ones in `reference-path`.

#### PatchCore anomaly detection

```
python pyml-launch.py filesrc location=data/factory.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! pyml_anomaly device=cuda reference-path=data/factory_reference.npy threshold=0.5 \
  ! videoconvert ! autovideosink sync=false
```

### Audio Classification (CLAP)

`pyml_clap` classifies audio against the labels you pass with LAION CLAP.

#### CLAP audio event detection

```
python pyml-launch.py filesrc location=data/audio_sample.wav ! decodebin \
  ! audioconvert ! audioresample ! audio/x-raw,format=F32LE,rate=48000,channels=1 \
  ! pyml_clap device=cuda labels="gunshot,siren,baby crying,music,speech" threshold=0.3 \
  ! fakesink
```

### Vision-Language Model (VLM)

`pyml_vlm` answers `prompt` about each frame with a vision-language model such as LLaVA or SmolVLM.

#### LLaVA visual question answering

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! pyml_vlm model-name=llava-hf/llava-1.5-7b-hf device=cuda \
            prompt="What is happening in this scene?" \
  ! fakesink
```

### Cascade Gating

Every video element takes `only-on`, the name of a blob a frame has to carry
before the element runs on it. `only-on=alert` runs on the frames `pyml_alert`
flagged, `only-on=detections` on the frames that carry at least one analytics
object, and any other value names the blob an element attached under its own
name (`depth`, `vlm`, ...). Frames without it pass through untouched, so a cheap
detector decides which frames an expensive model sees.

#### VLM only on alerted frames

The alert decides which frames the vision-language model sees. The same property
works on `pyml_caption_qwen`, `pyml_depth` or any other video element.

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! pyml_yolo model-name=yolo11m device=cuda ! pyml_alert rules='{"class":"person","min_score":0.8}' cooldown=5 draw-alert=false ! pyml_vlm model-name=HuggingFaceTB/SmolVLM-500M-Instruct device=cuda only-on=alert max-tokens=40 prompt="What is the person in the centre doing?" ! pyml_metasink location=people.jsonl
```

The g2g host hands a hosted element the frame's upstream detections and
blobs, so `only-on` gates the same way on both backends.

### Embedding Extractor

`pyml_embedding` attaches a CLIP or DINOv2 embedding to each frame.

#### CLIP embedding extraction

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! pyml_embedding model-name=openai/clip-vit-base-patch32 device=cuda \
            normalize=true \
  ! fakesink
```

#### DINOv2 embeddings saved to file

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! pyml_embedding model-name=facebook/dinov2-base device=cuda \
            frame-stride=5 \
  ! fakesink
```

### Video Memory

`pyml_embeddingsink` stores the vectors `pyml_embedding` produces in an sqlite
index, one row per embedded frame: its `pts` in seconds, the `source-id` you give
the stream, and the embedding. Search it by text through the MCP server's
`search_video`. One index holds one model, and the embedding blob does not carry
the model name, so set `model-name` on the sink to the extractor's model.

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! pyml_embedding model-name=openai/clip-vit-base-patch32 device=cuda frame-stride=30 ! pyml_embeddingsink location=people.sqlite source-id=people model-name=openai/clip-vit-base-patch32
```

### Multi-Object Tracker

`pyml_tracker` tracks the detections of any upstream detector.

#### YOLO + standalone SORT tracker

```
python pyml-launch.py filesrc location=data/soccer_tracking.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! pyml_objectdetector model-name=fasterrcnn_resnet50_fpn device=cuda \
  ! pyml_tracker tracker-type=sort max-age=30 min-hits=3 iou-threshold=0.3 \
  ! pyml_overlay ! videoconvert ! autovideosink sync=false
```

### ML Alert

`pyml_alert` fires on upstream detections that match `rules`.

#### Webhook alert on person detection

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! pyml_objectdetector model-name=fasterrcnn_resnet50_fpn device=cuda \
  ! pyml_alert rules='{"class":"person","min_score":0.8}' \
              webhook-url=http://localhost:8080/alert cooldown=10 \
  ! pyml_overlay ! videoconvert ! autovideosink sync=false
```

#### MQTT alert with zone filtering

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! pyml_yolo model-name=yolo11m device=cuda \
  ! pyml_alert rules='{"class":"person","min_score":0.7,"zone":[0,0,320,240]}' \
              mqtt-broker=localhost:1883 mqtt-topic=alerts/zone1 cooldown=5 \
  ! pyml_overlay ! videoconvert ! autovideosink sync=false
```

### Alert Recorder

`pyml_alertrecorder` writes a clip around each alert that an upstream `pyml_alert`
attached to the buffer. It keeps `seconds-before` of video in memory and records
until `seconds-after` have passed since the last alert. `%s` in `location` is the
time the alert fired. The default encoder is `vp8enc deadline=1 ! webmmux`. Set
`encoder` and the extension together, for example
`encoder="x264enc tune=zerolatency ! mp4mux" location=alert-%s.mp4`.

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! pyml_yolo model-name=yolo11m device=cuda ! pyml_alert rules='{"class":"person","min_score":0.7}' cooldown=8 draw-alert=false ! pyml_alertrecorder location=alert-%s.webm seconds-before=2 seconds-after=3 ! pyml_overlay ! videoconvert ! autovideosink sync=false
```

### Metadata Sink

`pyml_metasink` writes one JSON line per buffer with whatever the pipeline knows
about it: `pts` in seconds, `detections` from the analytics metadata, any JSON
blob an element attached under its name (`alert`, `depth`, `vlm`, ...) and, for a
text stream, `text`. Lines go to `location`, or to stdout when it is unset. Use a
`tee` to keep a display alongside it.

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! pyml_yolo model-name=yolo11m device=cuda ! pyml_alert rules='{"class":"person","min_score":0.7}' draw-alert=false ! pyml_metasink location=people.jsonl
```

```
{"pts": 0.08, "detections": [{"label": "stream_0_person", "x": 469, "y": 312, "w": 37, "h": 82, "score": 0.85}], "alert": [{"timestamp": 1789611763.6, "rule": {"class": "person", "min_score": 0.7}, "detection": {"label": "stream_0_person", "x": 469, "y": 312, "w": 37, "h": 82, "score": 0.85}}]}
```

### Metadata Replay

`pyml_metareplay` reads a JSON lines file a `pyml_metasink` wrote and reattaches
each record to the frame at the same timestamp: `detections` become analytics
metadata again, every other key becomes the blob it came from. A record matches a
frame whose timestamp is within half a frame duration of it. Downstream
`pyml_tracker`, `pyml_alert`, `pyml_overlay` and `pyml_alertrecorder` then run
with no model loaded. Record once, then rerun the rules as often as you like.

Record once:

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! pyml_yolo model-name=yolo11m device=cuda confidence=0.5 ! pyml_metasink location=people.jsonl
```

Then replay. `data/people.jsonl` is that recording, shipped so this line runs with
no GPU at all:

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! pyml_metareplay location=data/people.jsonl ! pyml_tracker tracker-type=sort ! pyml_overlay ! videoconvert ! autovideosink sync=false
```

## MCP Server

`pyml-mcp` is an MCP server over stdio that runs the gst backend in-process, so an
agent can start a pipeline, read its results and change a property while it runs.
End every pipeline with a `pyml_metasink`, that is where records come from.

- `start_pipeline`, `pipeline_status`, `stop_pipeline`, `set_property`, `get_property`, `list_elements`, `inspect`.
- `latest_metadata` returns the newest records. `wait_for_records` blocks until the sink posts new ones, optionally only those carrying a key such as `detections`.
- `load_metadata` reads a JSON lines file a `pyml_metasink` wrote, so a finished run needs no pipeline.
- `snapshot_frame` returns the newest rendered frame as a JPEG. `describe_frame` captions it with the model `PYML_MCP_VLM_MODEL` names, `HuggingFaceTB/SmolVLM-500M-Instruct` by default.
- `search_video` reads an index `pyml_embeddingsink` wrote and returns the frames closest to a description, embedded with the index's own model. `clip_at` cuts a webm around a pts, which turns a hit into a clip.
- Every pipeline section of this README is a prompt named after it, such as `object_detection`.

Models run on `PYML_MCP_DEVICE`, `cuda` when torch sees one, else `cpu`, where a
caption takes tens of seconds.

```
uv sync --extra mcp
claude mcp add gst-python-ml -- /path/to/gst-python-ml/.venv/bin/pyml-mcp
```

For the g2g backend use glass2glass's own `g2g-mcp`, which drives its pipelines
natively. It has `latest_metadata`, `wait_for_records`, `load_metadata`,
`snapshot_frame`, `clip_at`, `set_property` and `get_property` under these names,
and prompts of its own. `describe_frame` and `search_video` stay here.
