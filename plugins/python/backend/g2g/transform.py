# BaseTransform (g2g backend)
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Library General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.

"""g2g backend for the `transform` element family.

The gst counterpart subclasses `GstBase.BaseTransform`; the g2g element is a
plain Python object driven by the host's `g2g_process(...)` call. All engine /
model logic still comes from the portable `MLEngineMixin`; the shared tunable
properties are declared here with the `GObject` shim so leaf code that reads
`self._<name>` is unchanged across backends.

The model is loaded lazily on the first frame (`_ensure_model`) rather than from
a `do_start` framework virtual, since the g2g host has no start hook.
"""

from backend.core import MLEngineMixin
from backend.g2g.shims import GObject


class BaseTransform(MLEngineMixin):
    """Base for g2g ML transform elements (same in/out format, e.g. detection)."""

    def __init__(self):
        self._ml_init()

    @GObject.Property(type=str)
    def device(self):
        "Device to run inference on (cpu, cuda, cuda:0, ...)"
        return self.mgr.device

    @device.setter
    def device(self, value):
        self.mgr.set_device(value)
        if self.mgr.engine_name:
            self.initialize_engine()

    @GObject.Property(type=int, default=1)
    def batch_size(self):
        "Number of items to process in a batch"
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value
        if self.engine:
            self.engine.batch_size = value

    @GObject.Property(type=int, default=1)
    def frame_stride(self):
        "How often to process a frame"
        return self._frame_stride

    @frame_stride.setter
    def frame_stride(self, value):
        self._frame_stride = value
        if self.engine:
            self.engine.frame_stride = value

    @GObject.Property(type=str)
    def model_name(self):
        "Name of the pre-trained model or local model path"
        return self._model_name

    @model_name.setter
    def model_name(self, value):
        self._model_name = value

    @GObject.Property(type=str)
    def engine_name(self):
        "ML engine to use: pytorch, onnx, tensorflow, tflite, openvino, ..."
        return self.mgr.engine_name

    @engine_name.setter
    def engine_name(self, value):
        self.mgr.engine_name = value

    @GObject.Property(type=str, default="auto")
    def input_format(self):
        "Input tensor layout: auto, nhwc, or nchw"
        return self.engine.input_format if self.engine else "auto"

    @input_format.setter
    def input_format(self, value):
        if self.engine:
            self.engine.input_format = value

    @GObject.Property(type=str, default="auto")
    def post_process(self):
        "Post-processing format for raw engine output"
        return self.engine.post_process if self.engine else "none"

    @post_process.setter
    def post_process(self, value):
        if self.engine:
            self.engine.post_process = value

    @GObject.Property(type=int, default=1)
    def device_queue_id(self):
        "ID of the DeviceQueue from the pool to use"
        return self._device_queue_id

    @device_queue_id.setter
    def device_queue_id(self, value):
        self._device_queue_id = value
        if self.engine:
            self.engine.device_queue_id = value

    @GObject.Property(type=bool, default=False)
    def compile(self):
        "Enable torch.compile optimization for the model"
        return self._compile

    @compile.setter
    def compile(self, value):
        self._compile = value
        if value:
            self.kwargs["compile"] = True
        else:
            self.kwargs.pop("compile", None)

    def _ensure_model(self):
        """Load the model on first use (the g2g host has no start hook).

        Guard on the model, not the engine: setting the `device` property
        eagerly creates the engine (with no model loaded), so an engine-only
        check would skip the load and leave inference with a null model.
        """
        if self.mgr.engine_name and (self.engine is None or self.engine.model is None):
            self.do_load_model()
