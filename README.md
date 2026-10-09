# GStreamer Python ML

[![CI](https://github.com/collabora/gst-python-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/collabora/gst-python-ml/actions/workflows/ci.yml)

Run any model with ease inside any GStreamer pipeline.

Pure Python ML elements for upstream GStreamer 1.24 or later: one element per task, or bring your own
model and run it with the base classes and matching engine. 

Write the pipeline in `gst-launch` format and run it with `pyml-launch`, with media backend set to either GStreamer or [glass2glass](https://gitlab.collabora.com/glass2glass/glass2glass).

## Features

- Video
  - object detection and zero-shot detection
  - tracking
  - pose estimation
  - depth estimation
  - zero-shot classification with CLIP or SigLIP
  - segmentation with SAM2
  - OCR
  - face detection and recognition
  - optical flow
  - super-resolution
  - action recognition
  - anomaly detection
  - captioning and vision-language models
  - embeddings and video search
- Audio
  - voice activity detection
  - transcription and translation
  - speech separation
  - text to speech
  - audio classification with CLAP
- Text
  - local and remote LLMs
  - incident digests
  - text to image
- Around them
  - alerts over webhooks and MQTT
  - clip recording
  - metadata sink and replay
  - Kafka sink
  - [MCP server](PIPELINES.md#mcp-server) for agents
- Engines
  - PyTorch by default
  - ONNX Runtime, TensorRT through ONNX Runtime, OpenVINO, LiteRT, TensorFlow
  - Apache TVM, tinygrad, Apple MLX, ExecuTorch, llama.cpp, JAX
  - MiGraphX, IREE, NCNN, Renesas DRP-AI, Rockchip RKNN, AMD VART
  - The task elements export their model once and run it on any engine with
    `engine-name=`, and on any accelerator with `device=`
  - CI runs every engine that installs from PyPI on the CPU and checks its
    output against PyTorch, on a YOLO export and on the task elements, see
    [Engine support](#engine-support)

## Engine support

Which task runs on which engine, as the parity tests check it on the CPU. A
footnote names what the engine or its converter refuses. The table is
generated from `plugins/python/engine/support_matrix.py`, regenerate it with
`python plugins/python/engine/support_matrix.py`.

<!-- support matrix start -->
| task | pytorch | onnx | openvino | tvm | tensorflow | tflite | ncnn | executorch | iree | tinygrad | migraphx | jax | llamacpp | mlx |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yolo | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | no | no | no |
| pose | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | no | no | no |
| depth | yes | yes | yes | yes | yes | yes | no [1] | yes | yes | yes | yes | yes | no | no |
| clip | yes | yes | yes | yes | partial [2] | partial [2] | partial [3] | yes | yes | yes | yes | yes | no | no |
| anomaly | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | no | no | no |
| embedding | yes | yes | yes | yes | yes | yes | partial [4] | yes | yes | yes | yes | no | no | no |
| action | yes | yes | yes | yes | yes | yes | no [5] | no [6] | yes | yes | yes | no | no | no |
| superres | yes | yes | yes | yes | no [7] | no [7] | yes | yes | yes | yes | yes | no | no | no |
| optical_flow | yes | yes | yes | yes | no [8] | no [8] | no [5] | yes | yes | yes | yes | no | no | no |
| sam | yes | yes | yes | yes | no [9] | no [9] | no [10] | yes | yes | yes | no [11] | no | no | no |
| zero_shot | yes | yes | yes | yes | no [12] | no [12] | no [12] | yes | yes | no [13] | no [12] | no | no | no |
| llm | yes | yes | yes | no | no | no | no | no | no | no | no | no | yes | yes |
| whisper | yes | yes | yes | no | no | no | no | no | no | no | no | no | no | no |

1. ncnn rejects the broadcast across the batch axis
2. google/siglip-base-patch16-224: onnx2tf splits a constant vector one off
3. google/siglip-base-patch16-224: ncnn rejects the reshape that indexes the batch
4. facebook/dinov2-small: ncnn rejects the broadcast across the batch axis
5. ncnn runs one frame at a time
6. executorch's convolution takes 3-d or 4-d input
7. onnx2tf slices a shape with a negative size
8. the onnx2tf raft flow is tens of pixels off
9. onnx2tf tiles a 4-d tensor with one multiple
10. ncnn cannot permute a 5-rank tensor
11. migraphx resizes only nearest and linear
12. converting owlv2 runs past 12 GB
13. tinygrad's Gather rejects 2-d constant indices
<!-- support matrix end -->

## Install

Ubuntu 24.04 or later:

```
sudo apt install -y python3-pip python3-venv \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-base-apps \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gir1.2-gst-plugins-bad-1.0 python3-gst-1.0 gstreamer1.0-python3-plugin-loader \
    libcairo2 libcairo2-dev git
```

Then a venv on the system Python, which is the one GStreamer's plugin loader
embeds. `uv sync` installs PyTorch, with CUDA on Linux, and the `yolo` extra
adds the YOLO models the quick start uses.

```
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/collabora/gst-python-ml.git && cd gst-python-ml
uv venv --python /usr/bin/python3 --system-site-packages
uv sync --extra yolo
export GST_PLUGIN_PATH=$PWD/plugins:$GST_PLUGIN_PATH
gst-inspect-1.0 python
```

The last line lists the `pyml_*` elements. [INSTALL.md](INSTALL.md) covers
Fedora, Windows, Docker, the feature extras, the other engines and custom
plugins.

## Quick start

Paths are relative to the checkout directory. Change `device=cuda` to `device=cpu` on a
machine without a GPU.

Track people with YOLO:

```
python pyml-launch.py filesrc location=data/soccer_tracking.mp4 ! decodebin ! videoconvertscale ! video/x-raw,width=640,height=480 ! pyml_yolo model-name=yolo11m device=cuda track=True ! pyml_overlay ! videoconvert ! autovideosink
```

Estimate depth with Depth Anything V2:

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! pyml_depth model-name=depth-anything/Depth-Anything-V2-Small-hf device=cuda ! videoconvert ! autovideosink sync=false
```

Transcribe Korean speech and translate it to English. Transcripts are logged at
GStreamer info level:

```
GST_DEBUG=python:4 python pyml-launch.py filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! pyml_whispertranscribe device=cuda language=ko translate=yes ! fakesink
```

Record a clip around every person detection:

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! pyml_yolo model-name=yolo11m device=cuda ! pyml_alert rules='{"class":"person","min_score":0.7}' cooldown=8 draw-alert=false ! pyml_alertrecorder location=alert-%s.webm seconds-before=2 seconds-after=3 ! pyml_overlay ! videoconvert ! autovideosink sync=false
```

Replay a recorded detection stream through the tracker and overlay, with no
model and no GPU:

```
python pyml-launch.py filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! pyml_metareplay location=data/people.jsonl ! pyml_tracker tracker-type=sort ! pyml_overlay ! videoconvert ! autovideosink sync=false
```

[PIPELINES.md](PIPELINES.md) describes every pipeline, one or two per element, how to
run them under glass2glass, and how to run the MCP server.

## Blog post

[![Unleashing gst-python-ml](https://www.collabora.com/assets/images/blog/Collabora-GStPython.jpg)](https://www.collabora.com/news-and-blog/blog/2025/05/12/unleashing-gst-python-ml-analytics-gstreamer-pipelines/)

**[Unleashing gst-python-ml: Python-powered ML analytics for GStreamer pipelines](https://www.collabora.com/news-and-blog/blog/2025/05/12/unleashing-gst-python-ml-analytics-gstreamer-pipelines/)**

Combining GStreamer with machine learning frameworks into video analytics
pipelines, from object detection to video captioning.
