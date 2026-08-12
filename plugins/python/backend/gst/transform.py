# BaseTransform (GStreamer backend)
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Library General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Library General Public License for more details.
#
# You should have received a copy of the GNU Library General Public
# License along with this library; if not, write to the
# Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
# Boston, MA 02110-1301, USA.

"""GStreamer backend for the `transform` element family (same input/output
format, e.g. object detection).

This file is the GStreamer half of the backend split: the element base
(`GstBase.BaseTransform`), the GObject property declarations, and the framework
virtuals (`do_start`, `do_set_caps`). All engine/model logic lives in the
portable `MLEngineMixin`; the property bodies just read/write its `self._*`
fields.
"""

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
from gi.repository import GObject, GstBase  # noqa: E402

from backend.core import MLEngineMixin  # noqa: E402


class BaseTransform(GstBase.BaseTransform, MLEngineMixin):
    """
    Base class for GStreamer transform elements that perform
    inference with a machine learning model. This class manages shared properties
    and handles model loading and device management via MLEngine.
    """

    __gstmetadata__ = (
        "BaseTransform",
        "Transform",
        "Generic machine learning model transform element",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    def __init__(self):
        super().__init__()
        self._ml_init()

    @GObject.Property(type=str)
    def device(self):
        "Device to run the inference on (cpu, cuda, cuda:0, cuda:1, etc.)"
        return self.mgr.device

    @device.setter
    def device(self, value):
        self.mgr.set_device(value)
        # todo why is this needed, for example for yolo ?
        if self.engine_name:
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
        "Machine Learning Engine to use : pytorch, tflite, tensorflow, onnx, openvino, tvm, tinygrad, mlx, executorch, llamacpp, candle, jax, or custom engine name"
        return self.mgr.engine_name

    @engine_name.setter
    def engine_name(self, value):
        self.mgr.engine_name = value

    @GObject.Property(type=str, default="auto")
    def input_format(self):
        "Input tensor layout: auto, nhwc, or nchw"
        if self.engine:
            return self.engine.input_format
        return "auto"

    @input_format.setter
    def input_format(self, value):
        if self.engine:
            self.engine.input_format = value

    @GObject.Property(type=str, default="auto")
    def post_process(self):
        "Post-processing format for raw engine output (auto, none, or a key from detection_decoder)"
        if self.engine:
            return self.engine.post_process
        return "none"

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

    @GObject.Property(type=bool, default=False, nick="compile")
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

    # GStreamer framework virtual: load the model when the element starts, then
    # run whatever the element itself needs starting (the backend-neutral hook).
    def do_start(self):
        self.do_load_model()
        on_start = getattr(self, "on_start", None)
        if on_start:
            on_start()
        return True
