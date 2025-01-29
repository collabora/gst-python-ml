# GStreamer Python ML

This project provides Python base classes and GStreamer elements supporting a broad range
of ML features. 

Supported functionality includes:

1. object detection
1. tracking
1. video captioning
1. translation
1. transcription
1. speech to text
1. text to speech
1. text to image
1. LLMs
1. serializing model metadata to Kafka server

ML toolkits are supported via the `MLEngine` abstraction - we have nominal support for
TensorFlow, LiteRT and OpenVINO, but all testing thus far has been done with PyTorch.

These elements will work with your distribution's GStreamer packages. They have been tested on Ubuntu 24 with GStreamer 1.24.

## Python Version

All elements have been tested with Python 3.12, the installed version of Python on Ubuntu 24


## Install

Please refer to `INSTALL.md` for details

## Using GStreamer Python ML Elements

Please refer to `PIPELINES.md` for details