# GStreamer Python ML

[![CI](https://github.com/collabora/gst-python-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/collabora/gst-python-ml/actions/workflows/ci.yml)

Pure Python ML elements for upstream GStreamer 1.24 or later. Each task is one
element. Write the pipeline in `gst-launch` spelling and run it with
`pyml-launch`, on GStreamer or on
[glass2glass](https://gitlab.collabora.com/glass2glass/glass2glass).

## Features

- **Video**: object detection, zero-shot detection, tracking, pose, depth,
  zero-shot classification with CLIP or SigLIP, segmentation with SAM2, OCR,
  face recognition, optical flow, super-resolution, action recognition, anomaly
  detection, captioning, vision-language models and embeddings.
- **Audio**: voice activity detection, transcription, translation, speech
  separation, text to speech and audio classification with CLAP.
- **Text**: local and remote LLMs, digests, text to image.
- **Around them**: alerts over webhooks and MQTT, clip recording, metadata sinks
  and replay, a Kafka sink, and an [MCP server](PIPELINES.md#mcp-server) for
  agents.
- **Engines**: PyTorch by default. Also ONNX Runtime, OpenVINO, LiteRT,
  TensorFlow, Apache TVM, tinygrad, Apple MLX, ExecuTorch, llama.cpp, Candle,
  JAX, MiGraphX, IREE, NCNN and Renesas DRP-AI. CI runs every engine that
  installs from PyPI on the CPU and checks its output against PyTorch.

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
embeds. `uv sync` installs PyTorch, with CUDA on Linux.

```
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/collabora/gst-python-ml.git && cd gst-python-ml
uv venv --python /usr/bin/python3 --system-site-packages
uv sync
export GST_PLUGIN_PATH=$PWD/plugins:$GST_PLUGIN_PATH
gst-inspect-1.0 python
```

The last line lists the `pyml_*` elements. [INSTALL.md](INSTALL.md) covers
Fedora, Windows, Docker, the other engines and custom plugins.

## Quick start

Paths are relative to the checkout. Change `device=cuda` to `device=cpu` on a
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

[PIPELINES.md](PIPELINES.md) has every pipeline, one or two per element, how to
run them under glass2glass, and the MCP server.

## Blog post

[Unleashing gst-python-ml: Analytics GStreamer Pipelines](https://www.collabora.com/news-and-blog/blog/2025/05/12/unleashing-gst-python-ml-analytics-gstreamer-pipelines/)
walks through the elements from object detection to video captioning.
