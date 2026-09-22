# Install

The short path is in the [README](README.md#install). This file has the rest: Fedora, Windows, Docker, the engine extras and custom plugins.

## Host Install

### Install distribution packages

#### Ubuntu
```
sudo apt update && sudo apt -y upgrade
sudo apt install -y python3-pip  python3-venv \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-base-apps \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gir1.2-gst-plugins-bad-1.0 python3-gst-1.0 gstreamer1.0-python3-plugin-loader \
    libcairo2 libcairo2-dev git
```

#### Fedora

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



#### Windows

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

### Manage Python packages

#### Python version

GStreamer's Python plugin loader embeds the system interpreter, so the venv must
use the same Python. Ubuntu 24.04 is 3.12, Fedora 42 and Ubuntu 26.04 are 3.14. A
mismatch shows up at run time as `No module named 'torch'`.

#### venv on the system Python

```
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[yolo]"
```

#### With uv

Point uv at the system Python, not a downloaded one:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python /usr/bin/python3 --system-site-packages
source .venv/bin/activate
uv sync --extra yolo
```

Do not pre-install torch from the PyTorch index here. `uv sync` resolves torch from
`uv.lock`, which on Linux already pulls the CUDA wheels, and replaces whatever was
installed before.

### Feature extras

The core install covers PyTorch, torchvision, transformers and OpenCV, which is
what most elements need. Each extra adds the packages for a group of elements:

- `yolo`: `pyml_yolo`, `pyml_yolo_pose`, `demo_soccer`, and the `bytetrack` and `botsort` types of `pyml_tracker`
- `llm`: `pyml_llm`, `pyml_llmstreamfilter`, `pyml_caption_qwen`, `pyml_caption_phi`
- `awq`: AWQ quantized models, through gptqmodel, which builds its CUDA kernels with nvcc on the first model load
- `audio`: `pyml_whisperspeechtts`, `pyml_whisperlive`, `pyml_demucs`, `pyml_sepformer`
- `diffusion`: `pyml_stablediffusion`
- `face`: `pyml_face`
- `superres`: `pyml_superres`
- `kafka`: `pyml_kafkasink`
- `mqtt`: `pyml_alert` with `mqtt-broker=` set
- `mcp`: the MCP server
- `vad`: `pyml_vad`

```
uv sync --extra yolo --extra audio
```

`all` covers `yolo`, `llm`, `audio`, `diffusion`, `face`, `superres`, `kafka`, `mqtt`
and `vad`, plus the ONNX, tinygrad, llama.cpp, OpenVINO, TensorFlow and LiteRT
engines. It leaves out `awq`, which needs nvcc. TensorFlow has no 3.14 wheel
either, so `all` resolves on 3.12 and 3.13 only.

### Engine extras

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

### flash-attn

Install the prebuilt wheel matching your Python, torch and CUDA from
[flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases), for example:

```
pip install ./flash_attn-2.8.3+cu128torch2.11-cp314-cp314-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
```

### MiGraphX (AMD ROCm)

#### Ubuntu

Needs ROCm:
```
sudo apt install migraphx
```

Make the `migraphx` module importable:
```
export PYTHONPATH=/opt/rocm/lib:$PYTHONPATH
```

#### Fedora

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

### Clone repo

```
cd $HOME/src
git clone https://github.com/collabora/gst-python-ml.git
```

### Plugin path

```
echo 'export GST_PLUGIN_PATH=$HOME/src/gst-python-ml/plugins:$GST_PLUGIN_PATH' >> ~/.bashrc
source ~/.bashrc
```

## Docker Install

The Dockerfiles mount the checkout from `$HOME/src/gst-python-ml`.

### Enable Docker GPU Support on Host

Skip this on CPU.


#### Ubuntu
```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### Fedora

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


### Build Container

`docker build -f ./Dockerfile_ubuntu24 -t ubuntu24:latest .`

`docker build -f ./Dockerfile_ubuntu26 -t ubuntu26:latest .`

`docker build -f ./Dockerfile_fedora42 -t fedora42:latest .`


### Run Docker Container

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
