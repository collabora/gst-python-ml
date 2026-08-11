# GStreamer Python ML

[![CI](https://github.com/collabora/gst-python-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/collabora/gst-python-ml/actions/workflows/ci.yml)

This project provides a pure Python ML framework for upstream GStreamer, supporting a broad range of ML vision and language features. 

Supported functionality includes:

1. object detection
1. tracking
1. pose estimation (COCO 17-keypoint skeleton)
1. monocular depth estimation
1. zero-shot classification (CLIP / SigLIP)
1. video captioning
1. translation
1. transcription
1. voice activity detection
1. speech to text
1. text to speech
1. text to image
1. LLMs
1. serializing model metadata to Kafka server

Different ML toolkits are supported via the `MLEngine` abstraction: PyTorch, ONNX Runtime, OpenVINO,
LiteRT (TFLite), TensorFlow, Apache TVM, tinygrad, Apple MLX, Meta ExecuTorch, llama.cpp, HuggingFace Candle, JAX/Flax, AMD MiGraphX, IREE, and NCNN (Vulkan).
All testing thus far has been done primarily with PyTorch.

These elements will work with your distribution's GStreamer packages as long as the GStreamer version
is >= 1.24.

## Table of Contents

- [Install](#install)
  - [Host Install](#host-install)
  - [Docker Install](#docker-install)
- [Post Install](#post-install)
- [Custom Plugins](#custom-plugins)
- [Pipelines](#pipelines)
  - [Classification](#classification)
  - [Torch Compile](#torch-compile)
  - [Object Detection](#object-detection)
  - [Pose Estimation](#pose-estimation)
  - [Depth Estimation](#depth-estimation)
  - [Zero-Shot Classification (CLIP / SigLIP)](#zero-shot-classification-clip--siglip)
  - [Voice Activity Detection](#voice-activity-detection)
  - [Transcription](#transcription)
  - [LLM](#llm)
  - [Remote LLM (Ollama)](#remote-llm-ollama)
  - [Stable Diffusion](#stablediffusion)
  - [Kafka Sink](#kafkasink)
  - [Segment Anything (SAM)](#segment-anything-sam)
  - [OCR](#ocr)
  - [Face Detection & Recognition](#face-detection--recognition)
  - [Optical Flow](#optical-flow)
  - [Super-Resolution](#super-resolution)
  - [Action Recognition](#action-recognition)
  - [Anomaly Detection](#anomaly-detection)
  - [Audio Classification (CLAP)](#audio-classification-clap)
  - [Vision-Language Model (VLM)](#vision-language-model-vlm)
  - [Embedding Extractor](#embedding-extractor)
  - [Multi-Object Tracker](#multi-object-tracker)
  - [ML Alert](#ml-alert)

## Install

There are two installation options described below: on host machine or on Docker container:

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

(adjust Fedora version from 42 to match your version number)

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

1. **Install GStreamer** from the [official site](https://gstreamer.freedesktop.org/download/#windows).
   Download and install both the **runtime** and **development** MSVC x86_64 installers.
   The default install path is `C:\gstreamer\1.0\msvc_x86_64`.

2. **Set environment variables** (adjust paths if your install location differs):

```powershell
# Add GStreamer to PATH
[Environment]::SetEnvironmentVariable("PATH", "C:\gstreamer\1.0\msvc_x86_64\bin;" + $env:PATH, "User")

# Point GStreamer at your plugin directory
[Environment]::SetEnvironmentVariable("GST_PLUGIN_PATH", "D:\Workspace\gst-python-ml\plugins", "User")
```

3. **Install Python 3.12+** from [python.org](https://www.python.org/downloads/) or via conda.

4. **Install PyGObject** — on Windows the easiest route is via conda or the
   [gstreamer-python](https://pypi.org/project/gstreamer-python/) wheel:

```powershell
pip install gstreamer-python
```

5. **CUDA (optional)** — install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
   matching your GPU driver version, then install the CUDA-enabled PyTorch:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

#### Manage Python packages

##### Important: Python version must match GStreamer

GStreamer's Python plugin loader (`libgstpython.so`) embeds the system Python interpreter.
The virtual environment **must** be created with the same Python version that GStreamer uses,
otherwise `import` errors will occur at runtime (e.g. `No module named 'torch'`).

On Fedora 42+ this is Python 3.14. On Ubuntu 26.04 this is Python 3.14.
On Ubuntu 24.04 this is Python 3.12.

##### set up venv with system Python

```
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

##### Alternative: manage with uv

If using uv, ensure uv uses the **system** Python (not a downloaded one):

```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python /usr/bin/python3 --system-site-packages
source .venv/bin/activate
uv sync
```

Do not pre-install torch from the PyTorch CUDA index here. `uv sync` resolves torch
from `uv.lock` (PyPI), which on Linux already pulls the CUDA wheels, and it will
replace anything installed beforehand.

#### ONNX Runtime

For CPU inference:
```
uv sync --extra onnx
```

For GPU inference (requires CUDA):
```
uv sync --extra onnx-gpu
```

#### tinygrad

```
pip install tinygrad
```
or
```
uv sync --extra tinygrad
```

#### Apple MLX (macOS Apple Silicon only)

```
pip install mlx mlx-lm
```
or
```
uv sync --extra mlx
```

#### ExecuTorch

Requires Python 3.10–3.13 (no 3.14 wheel yet).

```
pip install executorch
```
or
```
uv sync --extra executorch
```

#### llama.cpp

```
pip install llama-cpp-python
```
or
```
uv sync --extra llamacpp
```

For GPU support, set the build flag:
```
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
```

#### Candle

Candle (HuggingFace Rust inference) requires building from source with `maturin`:
```
pip install maturin
git clone https://github.com/huggingface/candle.git
cd candle/candle-pyo3
maturin develop -r
```

#### Apache TVM

TVM is a deep learning compiler for model optimization and deployment. The PyPI
`apache-tvm` package is stale — install from source:

```
sudo apt install zlib1g-dev libxml2-dev  # Ubuntu/Debian
git clone --recursive https://github.com/apache/tvm.git
cd tvm
mkdir build && cd build
cp ../cmake/config.cmake .
echo "set(CMAKE_BUILD_TYPE RelWithDebInfo)" >> config.cmake
echo "set(USE_LLVM \"llvm-config --ignore-libllvm --link-static\")" >> config.cmake
echo "set(USE_CUDA ON)" >> config.cmake  # set OFF if no GPU
cmake .. && cmake --build . --parallel $(nproc)
cd ../3rdparty/tvm-ffi && pip install . && cd ../..
pip install -e .
```

Requires: CMake >= 3.24, LLVM >= 15, Python >= 3.10.
See [TVM install docs](https://tvm.apache.org/docs/install/from_source.html) for full details.

#### JAX

For CPU:
```
pip install jax[cpu]
```
or
```
uv sync --extra jax-cpu
```

For GPU (CUDA 12):
```
pip install jax[cuda12]
```
or
```
uv sync --extra jax-gpu
```

Now manually install flash-attn wheel (must match your version of python, torch and cuda)
For example, for torch 2.11 + CUDA 12.8 + Python 3.14:

`pip install ./flash_attn-2.8.3+cu128torch2.11-cp314-cp314-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl`

Pre-built wheels can be found here:
https://github.com/mjun0812/flash-attention-prebuild-wheels/releases


#### MiGraphX (AMD ROCm)

MiGraphX is AMD's graph inference engine for optimized model execution on AMD GPUs (ROCm).

Install MiGraphX (requires ROCm):
```
sudo apt install migraphx
```

Set the Python path so that the `migraphx` module is importable:
```
export PYTHONPATH=/opt/rocm/lib:$PYTHONPATH
```

#### IREE

IREE is a compiler-based ML runtime (backed by AMD/Google) supporting ROCm, Vulkan, CUDA, and CPU.

```
pip install iree-base-compiler[onnx] iree-base-runtime
```

#### NCNN (Vulkan)

NCNN is a lightweight inference framework with Vulkan GPU support for AMD/NVIDIA/Intel GPUs
(no ROCm or CUDA required).

```
pip install ncnn
```

Convert ONNX models to NCNN format:
```
pip install onnx-simplifier
python -m onnxsim model.onnx model_sim.onnx
# Use ncnn's onnx2ncnn tool:
onnx2ncnn model_sim.onnx model.param model.bin
```

#### AMD Ryzen AI (NPU)

For AMD Ryzen AI laptops (7040/8040/Strix Point) with on-chip NPU, use the ONNX engine
with `device=npu`. Requires the [Ryzen AI SDK](https://ryzenai.docs.amd.com/):

```
# Install Ryzen AI SDK (provides VitisAIExecutionProvider for ONNX Runtime)
# See: https://ryzenai.docs.amd.com/en/latest/inst.html
```

#### AMD ZenDNN (CPU Optimization)

ZenDNN optimizes inference on AMD EPYC/Zen CPUs. It works as a backend for existing
frameworks (ONNX Runtime, TensorFlow) — no separate engine needed. Set environment
variables to enable:

```
export ZENDNN_INT8_SUPPORT=1
export OMP_NUM_THREADS=$(nproc)
# Then use engine-name=onnx or engine-name=tensorflow as usual
```

See: https://www.amd.com/en/developer/zendnn.html

#### Clone repo

```
cd $HOME/src
git clone https://github.com/collabora/gst-python-ml.git
```

#### Update .bashrc

```
echo 'export GST_PLUGIN_PATH=$HOME/src/gst-python-ml/plugins:$GST_PLUGIN_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Docker Install

#### Build Docker Container

Important Note:

This Dockerfile maps a local `gst-python-ml` repository to the container,
and expects this repository to be located in `$HOME/src` i.e.  `$HOME/src/gst-python-ml`.


#### Enable Docker GPU Support on Host

To use the host GPU in a docker container, you will need to install the nvidia container toolkit. If running on CPU, these steps can be skipped.


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

Note: If running on CPU, just remove `--gpus all` from commands below:

`docker run -v ~/src/gst-python-ml/:/root/gst-python-ml -it --rm --gpus all --name ubuntu24 ubuntu24:latest /bin/bash`

or

`docker run -v ~/src/gst-python-ml/:/root/gst-python-ml -it --rm --gpus all --name ubuntu26 ubuntu26:latest /bin/bash`

or

`docker run -v ~/src/gst-python-ml/:/root/gst-python-ml -it --rm --gpus all --name fedora42 fedora42:latest /bin/bash`

Now, in the container shell, set up the `venv` as detailed above.


## Post Install

Run `gst-inspect-1.0 python` to list pyml elements.

## Custom Plugins

You can create your own GStreamer elements that inherit from the gst-python-ml base classes
(`BaseObjectDetector`, `BaseTransform`, `BaseClassifier`, etc.) in a separate directory.

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

Note: When a pipeline begins, GStreamer scans all scripts for GStreamer elements, including elements that are not actually in the pipeline. To ensure that startup
is fast, please avoid placing heavy imports such as NumPy at the module level, as these
will be imported by GStreamer. Instead, favour importing at the method level - since Python caches imports, this will have no performance impact.


### Environment Setup

Set both `GST_PLUGIN_PATH` (so GStreamer discovers your `.py` files) and `PYTHONPATH`
(so Python can import your modules):

```bash
export GST_PLUGIN_PATH=$HOME/src/gst-python-ml/plugins:$HOME/my_plugins:$GST_PLUGIN_PATH
export PYTHONPATH=$HOME/my_plugins/python:$PYTHONPATH
```

The gst-python loader adds the first `python/` directory it finds to `sys.path`.
By listing the framework directory first, all gst-python-ml base classes (`base_objectdetector`,
`base_transform`, `base_classifier`, `base_caption`, `base_llm`, etc.) are importable
by custom plugins. The `PYTHONPATH` entry ensures gst-python can also resolve your
custom modules from the second directory.

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

## Using GStreamer Python ML Elements

## Running a pipeline on either backend

The ML elements run under GStreamer or under [glass2glass](https://gitlab.collabora.com/glass2glass/glass2glass),
selected by `GSTML_BACKEND`. `pyml-launch` takes one pipeline in `gst-launch`
spelling and runs it on whichever is selected:

```bash
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale \
  ! video/x-raw,width=640,height=480 \
  ! pyml_yolo model-name=yolo11m device=cuda:0 track=True \
  ! pyml_overlay ! videoconvert ! autovideosink

GSTML_BACKEND=g2g python pyml-launch.py <the same pipeline>
```

Run it from the checkout with any Python: it re-runs itself under `.venv` if it
finds one, so the elements see torch and the rest. Installing the package also
puts a `pyml-launch` on `PATH`.

Under `gst` it calls `gst-launch-1.0` with `GST_PLUGIN_PATH` pointing at this
checkout. Under `g2g` it calls `g2g-launch-py`, and first rewrites the three
things g2g spells differently: a `pyml_*` element becomes `pyelement` plus the
module and class to host, `pyml_overlay` becomes g2g's native
`analyticsoverlay`, and a raw-video caps filter with no format gains
`format=RGBA`. An element it cannot map is an error, not a silent pass-through.

`analyticsoverlay` has its own properties (`show-label`, `show-track`,
`show-score`, `show-trail`, `trail-length`, `thickness`, `mask-alpha`), so
`pyml_overlay`'s do not carry over; pass none and set them on the g2g side.

Build `g2g-launch-py` from a glass2glass checkout and put it on `PATH`, or point
`G2G_LAUNCH` at it:

```bash
PYO3_PYTHON=$(which python) cargo build --release -p g2g-python --features ml \
  --bin g2g-launch-py
```

## Pipelines

Below are some sample pipelines for the various elements in this project.

### Classification

```
GST_DEBUG=4 gst-launch-1.0  filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! pyml_classifier model-name=resnet18 device=cuda !  videoconvert !  autovideosink
```


### Torch Compile

Any PyTorch element supports `torch.compile` optimization via the `compile` property.
This triggers PyTorch's JIT compiler to fuse operations and generate optimized kernels,
improving steady-state throughput at the cost of a longer first-frame warm-up.

#### Classification with torch.compile

```
GST_DEBUG=4 gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale \
  ! video/x-raw,width=640,height=480 \
  ! pyml_classifier model-name=resnet18 device=cuda compile=True \
  ! videoconvert ! autovideosink
```

#### Object detection with torch.compile

```
GST_DEBUG=4 gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale \
  ! video/x-raw,width=640,height=480 \
  ! pyml_objectdetector model-name=fasterrcnn_resnet50_fpn device=cuda compile=True \
  ! videoconvert ! pyml_overlay ! videoconvert ! autovideosink
```


### Object Detection

#### TorchVision

`pyml_objectdetector` supports all TorchVision  object detection models.
Simply choose a suitable model name and set it on the `model-name` property.
A few possible model names:

```
fasterrcnn_resnet50_fpn
ssdlite320_mobilenet_v3_large
```

##### fasterrcnn

`GST_DEBUG=4 gst-launch-1.0  filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! pyml_objectdetector model-name=fasterrcnn_resnet50_fpn device=cuda batch-size=4 ! videoconvert ! pyml_overlay ! videoconvert ! autovideosink`

##### fasterrcnn/kafka

a) run pipeline from host

```
GST_DEBUG=4 gst-launch-1.0  filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! pyml_objectdetector model-name=fasterrcnn_resnet50_fpn device=cuda batch-size=4 ! pyml_kafkasink schema-file=data/pyml_object_detector.json broker=localhost:29092 topic=test-kafkasink-topic
```

b) run pipeline from docker

```
GST_DEBUG=4 gst-launch-1.0  filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! pyml_objectdetector model-name=fasterrcnn_resnet50_fpn device=cuda batch-size=4 ! pyml_kafkasink schema-file=data/pyml_object_detector.json broker=kafka:9092 topic=test-kafkasink-topic
```


#### maskrcnn

```
GST_DEBUG=4 gst-launch-1.0   filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! pyml_maskrcnn device=cuda batch-size=4 model-name=maskrcnn_resnet50_fpn ! videoconvert ! pyml_overlay ! videoconvert ! autovideosink
```

#### yolo with tracking

```
GST_DEBUG=4 gst-launch-1.0   filesrc location=data/soccer_tracking.mp4 ! decodebin !  videoconvertscale ! video/x-raw,width=640,height=480 ! pyml_yolo model-name=yolo11m device=cuda:0 track=True ! pyml_overlay  ! videoconvert ! autovideosink
```

```
GST_DEBUG=4 gst-launch-1.0   filesrc location=data/soccer_tracking.mp4 ! decodebin ! videoconvertscale ! video/x-raw,width=640,height=480,format=RGB ! pyml_streammux name=mux   filesrc location=data/soccer_tracking.mp4 ! decodebin ! videoconvertscale ! video/x-raw,width=640,height=480,format=RGB ! mux.   mux. ! pyml_yolo model-name=yolo11m device=cuda:0 track=True ! pyml_streamdemux name=demux   demux. ! queue ! videoconvert ! pyml_overlay ! videoconvert ! autovideosink sync=false   demux. ! queue ! videoconvert ! pyml_overlay ! videoconvert !  autovideosink sync=false

```

```
GST_DEBUG=4 gst-launch-1.0 filesrc location=data/soccer_tracking.mp4 ! decodebin ! videoconvertscale ! video/x-raw,width=640,height=480 ! demo_soccer model-name=yolo11m device=cuda:0 ! pyml_overlay ! videoconvert ! autovideosink
```


#### ONNX Engine

`pyml_objectdetector` supports any ONNX model via the `engine-name=onnx` property.
YOLO11 ONNX output (`[B, 4+nc, anchors]`) is automatically decoded with NMS — no manual post-processing required.

Export a YOLO11 model to ONNX with ultralytics:

```
yolo export model=yolo11m.pt format=onnx
```

##### YOLO11m ONNX object detection with overlay

Use `input-format=nchw` because YOLO expects channels-first input, and
`post-process=anchor_free` to decode the raw `[B, 4+nc, anchors]` output into
bounding boxes before handing off to `pyml_overlay`.

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! pyml_objectdetector engine-name=onnx model-name=yolo11m.onnx device=cpu \
              input-format=nchw post-process=anchor_free \
  ! videoconvert ! "video/x-raw,format=RGBA" \
  ! pyml_overlay ! videoconvert ! autovideosink
```

##### Generic ONNX passthrough (logs raw inference output)

Use `pyml_inference` to test any ONNX model and inspect raw output:

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! pyml_inference engine-name=onnx model-name=yolo11m.onnx device=cpu \
  ! fakesink
```

`pyml_inference` also accepts `engine-name=pytorch`, `engine-name=openvino`, etc.

#### OpenVINO Engine

Export a YOLO11 model to OpenVINO IR format with ultralytics:

```
yolo export model=yolo11m.pt format=openvino
```

This produces `yolo11m_openvino_model/yolo11m.xml` and `yolo11m.bin`.

##### YOLO11m OpenVINO object detection with overlay

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! pyml_objectdetector engine-name=openvino \
              model-name=yolo11m_openvino_model/yolo11m.xml device=cpu \
              input-format=nchw post-process=anchor_free \
  ! videoconvert ! "video/x-raw,format=RGBA" \
  ! pyml_overlay ! videoconvert ! autovideosink
```

Use `device=GPU` for Intel GPU acceleration (OpenVINO uses uppercase device names).

#### LiteRT (TFLite) Engine

Export a YOLO11 model to TFLite with ultralytics:

```
yolo export model=yolo11m.pt format=tflite
```

This produces `yolo11m_saved_model/yolo11m_float32.tflite`.

##### YOLO11m TFLite object detection with overlay

TFLite models expect NHWC input (default), so `input-format` does not need to be set.

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! pyml_objectdetector engine-name=tflite \
              model-name=yolo11m_saved_model/yolo11m_float32.tflite device=cpu \
              post-process=anchor_free \
  ! videoconvert ! "video/x-raw,format=RGBA" \
  ! pyml_overlay ! videoconvert ! autovideosink
```

#### TensorFlow Engine

Export a YOLO11 model to TensorFlow SavedModel with ultralytics:

```
yolo export model=yolo11m.pt format=saved_model
```

##### YOLO11m TensorFlow object detection with overlay

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! pyml_objectdetector engine-name=tensorflow \
              model-name=yolo11m_saved_model device=cuda \
              post-process=anchor_free \
  ! videoconvert ! "video/x-raw,format=RGBA" \
  ! pyml_overlay ! videoconvert ! autovideosink
```

#### tinygrad Engine

tinygrad supports TorchVision models, SafeTensors files, and Transformers models.
Set `engine-name=tinygrad` for lightweight GPU/CPU inference with automatic kernel optimization.

##### ResNet18 classification with tinygrad on GPU

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=224,height=224" \
  ! pyml_classifier model-name=resnet18 device=cuda engine-name=tinygrad \
  ! fakesink
```

##### tinygrad on CPU

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=224,height=224" \
  ! pyml_classifier model-name=resnet18 device=cpu engine-name=tinygrad \
  ! fakesink
```

#### TVM Engine

Apache TVM compiles models for optimized inference. Supports compiled `.so`/`.tar`
models and TorchVision models (auto-compiled via Relay). Set `engine-name=tvm`.

##### TorchVision model compiled with TVM

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=224,height=224" \
  ! pyml_classifier model-name=resnet18 device=cuda engine-name=tvm \
  ! fakesink
```

##### Pre-compiled TVM model (.so)

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! pyml_inference engine-name=tvm model-name=compiled_model.so device=cuda \
  ! fakesink
```

#### Apple MLX Engine

MLX is designed for Apple Silicon (M1/M2/M3/M4). Supports SafeTensors, `.npz` weights,
and mlx-lm text generation. Set `engine-name=mlx`.

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=224,height=224" \
  ! pyml_classifier model-name=resnet18 device=gpu engine-name=mlx \
  ! fakesink
```

#### ExecuTorch Engine

Meta ExecuTorch runs `.pte` models for on-device inference. Export a model with
`torch.export` + ExecuTorch, then set `engine-name=executorch`.

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=224,height=224" \
  ! pyml_inference engine-name=executorch model-name=model.pte device=cpu \
  ! fakesink
```

#### llama.cpp Engine

GGUF quantized LLM inference via llama-cpp-python. Set `engine-name=llamacpp`
and point to a `.gguf` model file.

```
gst-launch-1.0 filesrc location=data/prompt_for_llm.txt \
  ! pyml_llm engine-name=llamacpp model-name=model.gguf device=cpu \
  ! fakesink
```

#### Candle Engine

HuggingFace Candle (Rust) inference via Python bindings. Supports SafeTensors models.
Set `engine-name=candle`.

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=224,height=224" \
  ! pyml_inference engine-name=candle model-name=model.safetensors device=cpu \
  ! fakesink
```

#### JAX/Flax Engine

Google JAX with XLA compilation. Supports Flax checkpoints and HuggingFace models.
Set `engine-name=jax` for JIT-compiled inference on GPU, TPU, or CPU.

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=224,height=224" \
  ! pyml_classifier model-name=resnet18 device=cpu engine-name=jax \
  ! fakesink
```

#### MiGraphX Engine

AMD MiGraphX graph inference engine for optimized execution on AMD GPUs via ROCm.
Set `engine-name=migraphx` and point to an ONNX model file. Requires ROCm and
`migraphx` Python module (see install section above).

##### YOLO11m MiGraphX object detection with overlay

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! pyml_objectdetector engine-name=migraphx model-name=yolo11m.onnx device=gpu \
              input-format=nchw post-process=anchor_free \
  ! videoconvert ! "video/x-raw,format=RGBA" \
  ! pyml_overlay ! videoconvert ! autovideosink
```

##### MiGraphX on CPU (reference target)

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! pyml_inference engine-name=migraphx model-name=yolo11m.onnx device=cpu \
  ! fakesink
```

#### IREE Engine

IREE (Intermediate Representation Execution Environment) is a compiler-based ML runtime
backed by AMD and Google. Supports ROCm (AMD GPU), Vulkan, CUDA, and CPU targets.
Set `engine-name=iree` and point to a pre-compiled `.vmfb` or an `.onnx` model
(auto-compiled at load time).

##### IREE on AMD GPU (ROCm/HIP)

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! pyml_inference engine-name=iree model-name=yolo11m.onnx device=hip \
  ! fakesink
```

##### IREE on Vulkan (any GPU vendor)

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! pyml_inference engine-name=iree model-name=yolo11m.onnx device=vulkan \
  ! fakesink
```

##### IREE with pre-compiled module

```
# Pre-compile: iree-compile model.mlir --iree-hal-target-device=hip -o model.vmfb
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! pyml_inference engine-name=iree model-name=model.vmfb device=hip \
  ! fakesink
```

#### NCNN Engine (Vulkan)

NCNN is a lightweight inference framework with Vulkan GPU acceleration.
Works on AMD, NVIDIA, and Intel GPUs without requiring ROCm or CUDA.
Set `engine-name=ncnn` and point to an NCNN `.param` file (`.bin` must be alongside).

##### NCNN on Vulkan GPU

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! pyml_inference engine-name=ncnn model-name=yolo11m.param device=vulkan \
  ! fakesink
```

##### NCNN on CPU

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! pyml_inference engine-name=ncnn model-name=yolo11m.param device=cpu \
  ! fakesink
```

#### ONNX Runtime on AMD GPUs (ROCm)

The ONNX engine supports AMD GPUs via ROCm execution providers. Set `device=rocm`
to use MIGraphXExecutionProvider (preferred) or ROCMExecutionProvider as fallback.

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! pyml_objectdetector engine-name=onnx model-name=yolo11m.onnx device=rocm \
              input-format=nchw post-process=anchor_free \
  ! videoconvert ! "video/x-raw,format=RGBA" \
  ! pyml_overlay ! videoconvert ! autovideosink
```

#### ONNX Runtime on AMD Ryzen AI NPU

For AMD Ryzen AI laptops with on-chip NPU, set `device=npu`:

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! pyml_objectdetector engine-name=onnx model-name=yolo11m.onnx device=npu \
              input-format=nchw post-process=anchor_free \
  ! videoconvert ! "video/x-raw,format=RGBA" \
  ! pyml_overlay ! videoconvert ! autovideosink
```

#### PyTorch on AMD GPUs (ROCm)

The PyTorch engine supports AMD GPUs natively when PyTorch is installed with ROCm.
Use `device=cuda` (PyTorch uses the CUDA API mapping for ROCm):

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
```

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! pyml_objectdetector model-name=fasterrcnn_resnet50_fpn device=cuda \
  ! videoconvert ! pyml_overlay ! videoconvert ! autovideosink
```

With `torch.compile` and Triton for AMD GPU kernel optimization:

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! pyml_objectdetector model-name=fasterrcnn_resnet50_fpn device=cuda compile=True \
  ! videoconvert ! pyml_overlay ! videoconvert ! autovideosink
```

### Pose Estimation

`pyml_yolo_pose` supports all YOLO pose models. Recommended model names:
```
yolo11n-pose  (fastest)
yolo11s-pose
yolo11m-pose  (best accuracy)
```

#### YOLO pose with skeleton visualization (rendered on frame)

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue \
    ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
    ! pyml_yolo_pose model-name=yolo11n-pose device=cuda \
    ! videoconvert ! autovideosink sync=false
```

#### YOLO pose with bounding box overlay (metadata only, no in-element rendering)

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue \
    ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
    ! pyml_yolo_pose model-name=yolo11n-pose device=cuda visualize=false \
    ! videoconvert ! pyml_overlay ! videoconvert ! autovideosink sync=false
```

### Depth Estimation

`pyml_depth` supports DepthAnything V2 models from HuggingFace. Available model sizes:
```
depth-anything/Depth-Anything-V2-Small-hf  (fastest, ~100 MB)
depth-anything/Depth-Anything-V2-Base-hf
depth-anything/Depth-Anything-V2-Large-hf  (most accurate)
```

Available colormaps: `inferno` (default), `jet`, `viridis`, `plasma`, `magma`

#### DepthAnything V2 with inferno colormap

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue \
    ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
    ! pyml_depth model-name=depth-anything/Depth-Anything-V2-Small-hf device=cuda \
    ! videoconvert ! autovideosink sync=false
```

#### DepthAnything V2 with jet colormap

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue \
    ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
    ! pyml_depth model-name=depth-anything/Depth-Anything-V2-Small-hf device=cuda colormap=jet \
    ! videoconvert ! autovideosink sync=false
```

#### Depth with reduced compute via frame-stride

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue \
    ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
    ! pyml_depth model-name=depth-anything/Depth-Anything-V2-Small-hf device=cuda frame-stride=2 \
    ! videoconvert ! autovideosink sync=false
```

#### Depth with original video side-by-side (tee)

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue \
    ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
    ! tee name=t \
    t. ! queue ! pyml_depth model-name=depth-anything/Depth-Anything-V2-Small-hf device=cuda ! videoconvert ! autovideosink sync=false \
    t. ! queue ! videoconvert ! autovideosink sync=false
```

### Zero-Shot Classification (CLIP / SigLIP)

`pyml_clip` classifies each frame against a user-defined set of text labels
with no fixed label set — labels are set at pipeline launch time.

Supported models:
```
openai/clip-vit-base-patch32       (default, ~600 MB)
openai/clip-vit-large-patch14      (more accurate, ~1.7 GB)
google/siglip-base-patch16-224     (SigLIP, better zero-shot accuracy)
google/siglip-large-patch16-384    (SigLIP large)
```

#### CLIP with custom labels

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue \
    ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
    ! pyml_clip model-name=openai/clip-vit-base-patch32 device=cuda \
              labels="person, bicycle, car, dog, cat" top-k=3 \
    ! videoconvert ! pyml_overlay ! videoconvert ! autovideosink sync=false
```

#### SigLIP (better zero-shot accuracy than CLIP)

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue \
    ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
    ! pyml_clip model-name=google/siglip-base-patch16-224 device=cuda \
              labels="people walking, empty street, crowd, indoor scene" top-k=1 \
    ! videoconvert ! pyml_overlay ! videoconvert ! autovideosink sync=false
```

#### CLIP with threshold (only report labels above 20% confidence)

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue \
    ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
    ! pyml_clip model-name=openai/clip-vit-base-patch32 device=cuda \
              labels="person, bicycle, car, dog, cat" threshold=0.2 \
    ! videoconvert ! pyml_overlay ! videoconvert ! autovideosink sync=false
```

### Voice Activity Detection

#### Standalone VAD with metadata (pass-through, speech probability attached to buffers)

```
GST_DEBUG=4 gst-launch-1.0 pulsesrc ! audio/x-raw,format=S16LE,rate=16000,channels=1 ! pyml_vad threshold=0.7 ! fakesink
```

#### VAD gating before transcription (mute silent audio, reduce Whisper latency)

```
GST_DEBUG=4 gst-launch-1.0 filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! audioresample ! audio/x-raw,format=S16LE,rate=16000,channels=1 ! pyml_vad threshold=0.6 gate=true ! pyml_whispertranscribe device=cuda language=ko ! fakesink
```

### Transcription

#### transcription with initial prompt set

```
GST_DEBUG=4 gst-launch-1.0 filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! pyml_whispertranscribe device=cuda language=ko initial_prompt = "Air Traffic Control은, radar systems를,  weather conditions에, flight paths를, communication은, unexpected weather conditions가, continuous training을, dedication과, professionalism" ! fakesink
```

#### translation to English

```
GST_DEBUG=4 gst-launch-1.0 filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! pyml_whispertranscribe device=cuda language=ko translate=yes ! fakesink
```

#### demucs audio separation


```
GST_DEBUG=4 gst-launch-1.0 filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! audioresample ! pyml_demucs device=cuda ! wavenc ! filesink location=separated_vocals.wav
```


#### coquitts

```
GST_DEBUG=4 gst-launch-1.0 filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! pyml_whispertranscribe device=cuda language=ko translate=yes ! pyml_coquitts device=cuda ! audioconvert ! wavenc ! filesink location=output_audio.wav
```

#### whisperspeechtts

```
GST_DEBUG=4 gst-launch-1.0 filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! pyml_whispertranscribe device=cuda language=ko translate=yes ! pyml_whisperspeechtts device=cuda ! audioconvert ! wavenc ! filesink location=output_audio.wav
```

#### mariantranslate

```
GST_DEBUG=4 gst-launch-1.0 filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! pyml_whispertranscribe device=cuda language=ko translate=yes ! pyml_mariantranslate device=cuda src=en target=fr ! fakesink
```

Supported src/target languages:

https://huggingface.co/models?sort=trending&search=Helsinki


#### whisperlive

`GST_DEBUG=4 gst-launch-1.0 filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! pyml_whisperlive device=cuda language=ko translate=yes llm-model-name="microsoft/phi-2" ! audioconvert ! wavenc ! filesink location=output_audio.wav`

### LLM

1. generate HuggingFace token

2. `huggingface-cli login`
    and pass in token

3. LLM pipeline (in this case, we use phi-2)

`GST_DEBUG=4 gst-launch-1.0 filesrc location=data/prompt_for_llm.txt !  pyml_llm device=cuda model-name="microsoft/phi-2" ! fakesink`

#### Remote LLM

`pyml_llm_remote` sends text to a remote LLM endpoint via HTTP. The examples use
the OpenAI-compatible `/v1/chat/completions` path, which Ollama, llama.cpp and
vLLM all serve, so the same line works against any of them. The element picks the
request format from the URL: drop `url=` to fall back to its default, Ollama's
native `/api/generate`.

##### Basic call

```
GST_DEBUG=4 gst-launch-1.0 filesrc location=data/prompt_for_llm.txt \
  ! "text/x-raw,format=utf8" \
  ! pyml_llm_remote url=http://localhost:11434/v1/chat/completions \
    model-name=llama3 \
  ! fakesink
```

##### With a system prompt and a custom model

```
GST_DEBUG=4 gst-launch-1.0 filesrc location=data/prompt_for_llm.txt \
  ! "text/x-raw,format=utf8" \
  ! pyml_llm_remote url=http://localhost:11434/v1/chat/completions \
    model-name=qwen3:8b \
    system-prompt="You are a helpful assistant. Answer concisely." \
    temperature=0.5 \
  ! fakesink
```

### stablediffusion

`GST_DEBUG=4 gst-launch-1.0 filesrc location=data/prompt_for_stable_diffusion.txt ! pyml_stablediffusion device=cuda ! pngenc ! filesink location=output_image.png`

#### Caption

#### caption qwen with history

(should also work with "microsoft/Phi-3.5-vision-instruct" model)

```
GST_DEBUG=3 gst-launch-1.0 filesrc location=data/soccer_single_camera.mp4 ! decodebin ! videoconvertscale ! video/x-raw,width=640,height=480 ! tee name=t t. ! queue ! textoverlay name=overlay wait-text=false ! videoconvert ! autovideosink t. ! queue leaky=2 max-size-buffers=1 ! videoconvertscale ! video/x-raw,width=240,height=180 ! pyml_caption_qwen device=cuda:0 prompt="In one sentence, describe what you see?" model-name="Qwen/Qwen2.5-VL-3B-Instruct-AWQ" name=cap cap.src ! fakesink async=0 sync=0 cap.text_src ! queue ! coalescehistory history-length=10 ! pyml_llm model-name="Qwen/Qwen3-0.6B" device=cuda system-prompt="You receive the history of what happened in recent times, summarize it nicely with excitement but NEVER mention the specific times. Focus on the most recent events." ! queue ! overlay.text_sink
```

### kafkasink

#### Setting up kafka network

`docker network create kafka-network`

and list networks

`docker network ls`

#### docker launch

To launch a docker instance with the kafka network, add ` --network kafka-network  `
to the docker launch command above.

#### Set up kafka and zookeeper

Note: setup below assumes you are running your pipeline in a docker container. 
If running pipeline from host, then the port changes from `9092` to `29092`,
and the broker changes from `kafka` to `localhost`.

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

#### Create test topic
```
docker exec kafka kafka-topics --create --topic test-kafkasink-topic --bootstrap-server kafka:9092 --partitions 1 --replication-factor 1
```

#### list topics

`docker exec -it kafka kafka-topics --list --bootstrap-server kafka:9092`


#### delete topic

`docker exec -it kafka kafka-topics --delete --topic test-topic --bootstrap-server kafka:9092`


#### consume topic

`docker exec -it kafka kafka-console-consumer --bootstrap-server kafka:9092 --topic test-kafkasink-topic --from-beginning`


### non ML

`GST_DEBUG=4 gst-launch-1.0 videotestsrc ! video/x-raw,width=1280,height=720 ! pyml_overlay meta-path=data/sample_metadata.json tracking=true ! videoconvert ! autovideosink`


### streammux/streamdemux pipeline

```
 GST_DEBUG=4 gst-launch-1.0   videotestsrc pattern=ball ! video/x-raw, width=320, height=240 ! queue ! pyml_streammux name=mux   videotestsrc pattern=smpte ! video/x-raw, width=320, height=240 ! queue ! mux.sink_1   videotestsrc pattern=smpte ! video/x-raw, width=320, height=240 ! queue ! mux.sink_2   mux.src ! queue ! pyml_streamdemux name=demux   demux.src_0 ! queue ! glimagesink  demux.src_1 ! queue ! glimagesink   demux.src_2 ! queue  ! glimagesink
```

### Segment Anything (SAM)

`pyml_sam` runs Meta SAM2 for zero-shot segmentation with point, box, or automatic prompts.

#### Auto-mask segmentation (segment everything)

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! pyml_sam model-name=facebook/sam2-hiera-small device=cuda prompt-mode=auto \
  ! videoconvert ! autovideosink sync=false
```

#### Point-prompt segmentation (segment object at center)

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! pyml_sam model-name=facebook/sam2-hiera-small device=cuda \
            prompt-mode=point points="320,240" \
  ! videoconvert ! autovideosink sync=false
```

### OCR

`pyml_ocr` performs text detection and recognition using EasyOCR or TrOCR.

#### EasyOCR text detection (default)

```
gst-launch-1.0 filesrc location=data/document.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! pyml_ocr backend=easyocr languages="en" device=cuda \
  ! videoconvert ! pyml_overlay ! videoconvert ! autovideosink sync=false
```

#### TrOCR recognition

```
gst-launch-1.0 filesrc location=data/document.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! pyml_ocr backend=trocr model-name=microsoft/trocr-base-printed device=cuda \
  ! videoconvert ! pyml_overlay ! videoconvert ! autovideosink sync=false
```

### Face Detection & Recognition

`pyml_face` detects faces with RetinaFace and optionally identifies them using ArcFace embeddings.

#### Face detection only

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! pyml_face device=cuda \
  ! videoconvert ! pyml_overlay ! videoconvert ! autovideosink sync=false
```

#### Face detection + recognition with gallery

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! pyml_face device=cuda gallery-path=data/face_gallery/ recognition-threshold=0.6 \
  ! videoconvert ! pyml_overlay ! videoconvert ! autovideosink sync=false
```

### Optical Flow

`pyml_optical_flow` estimates dense optical flow between consecutive frames using RAFT.

#### RAFT optical flow with color visualization

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! pyml_optical_flow model-name=raft-small device=cuda visualize=true \
  ! videoconvert ! autovideosink sync=false
```

### Super-Resolution

`pyml_superres` upscales video frames using Real-ESRGAN.

#### 2x upscale

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=320,height=240" \
  ! pyml_superres device=cuda scale=2 \
  ! videoconvert ! autovideosink sync=false
```

#### 4x upscale with tile processing

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=320,height=240" \
  ! pyml_superres device=cuda scale=4 tile-size=256 tile-overlap=32 \
  ! videoconvert ! autovideosink sync=false
```

### Action Recognition

`pyml_action` classifies activities over sliding temporal windows using SlowFast or X3D.

#### SlowFast action recognition

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! pyml_action model-name=slowfast_r50 device=cuda clip-length=32 \
  ! videoconvert ! pyml_overlay ! videoconvert ! autovideosink sync=false
```

### Anomaly Detection

`pyml_anomaly` detects visual anomalies using PatchCore for manufacturing QA.

#### PatchCore anomaly detection

```
gst-launch-1.0 filesrc location=data/factory.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! pyml_anomaly device=cuda coreset-path=data/coreset.pt threshold=0.5 \
  ! videoconvert ! autovideosink sync=false
```

### Audio Classification (CLAP)

`pyml_clap` performs zero-shot audio classification using LAION CLAP.

#### CLAP audio event detection

```
gst-launch-1.0 filesrc location=data/audio_sample.wav ! decodebin \
  ! audioconvert ! audioresample ! audio/x-raw,format=F32LE,rate=48000,channels=1 \
  ! pyml_clap device=cuda labels="gunshot,siren,baby crying,music,speech" threshold=0.3 \
  ! fakesink
```

### Vision-Language Model (VLM)

`pyml_vlm` runs generic VLMs (LLaVA, InternVL, etc.) for visual question answering.

#### LLaVA visual question answering

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! pyml_vlm model-name=llava-hf/llava-1.5-7b-hf device=cuda \
            prompt="What is happening in this scene?" \
  ! fakesink
```

### Embedding Extractor

`pyml_embedding` extracts dense vector embeddings from video frames.

#### CLIP embedding extraction

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! pyml_embedding model-name=openai/clip-vit-base-patch32 device=cuda \
            output-mode=metadata \
  ! fakesink
```

#### DINOv2 embeddings saved to file

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! pyml_embedding model-name=facebook/dinov2-base device=cuda \
            output-mode=file output-path=embeddings.npy \
  ! fakesink
```

### Multi-Object Tracker

`pyml_tracker` is a standalone tracker that works with any upstream detector.

#### YOLO + standalone SORT tracker

```
gst-launch-1.0 filesrc location=data/soccer_tracking.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! pyml_objectdetector model-name=fasterrcnn_resnet50_fpn device=cuda \
  ! pyml_tracker tracker-type=sort max-age=30 min-hits=3 iou-threshold=0.3 \
  ! pyml_overlay ! videoconvert ! autovideosink sync=false
```

### ML Alert

`pyml_alert` triggers alerts based on upstream detection metadata.

#### Webhook alert on person detection

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! pyml_objectdetector model-name=fasterrcnn_resnet50_fpn device=cuda \
  ! pyml_alert rules='{"class":"person","min_score":0.8}' \
              webhook-url=http://localhost:8080/alert cooldown=10 \
  ! pyml_overlay ! videoconvert ! autovideosink sync=false
```

#### MQTT alert with zone filtering

```
gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! pyml_yolo model-name=yolo11m device=cuda \
  ! pyml_alert rules='{"class":"person","min_score":0.7,"zone":[0,0,320,240]}' \
              mqtt-broker=localhost:1883 mqtt-topic=alerts/zone1 cooldown=5 \
  ! pyml_overlay ! videoconvert ! autovideosink sync=false
```
```