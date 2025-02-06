# GstBaseTransform
# Copyright (C) 2024-2025 Collabora Ltd.
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

import gi
from engine.gst_engine_factory import GstEngineFactory

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
from gi.repository import Gst, GObject, GstBase  # noqa: E402

from log.logger_factory import LoggerFactory


class BatchBuffer:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.sub_buffers = []  # Store sub-buffers (frames)
        self.timestamps = []  # Store timestamps for each sub-buffer

    def add_sub_buffer(self, buf, pts):
        """
        Adds a sub-buffer (frame) to the batch buffer.
        """
        self.sub_buffers.append(buf)
        self.timestamps.append(pts)

    def get_sub_buffer(self, index):
        """
        Gets a specific sub-buffer by index.
        """
        return self.sub_buffers[index] if index < len(self.sub_buffers) else None

    def get_batch_size(self):
        """
        Returns the number of sub-buffers in the batch.
        """
        return len(self.sub_buffers)

    def get_min_timestamp(self):
        """
        Returns the minimum timestamp from the batch.
        """
        return min(self.timestamps) if self.timestamps else None

    def is_full(self):
        """
        Checks if the batch is full.
        """
        return self.get_batch_size() >= self.batch_size

    def clear(self):
        """
        Clears the batch buffer.
        """
        self.sub_buffers.clear()
        self.timestamps.clear()


class GstBaseTransform(GstBase.BaseTransform):
    """
    Base class for GStreamer transform elements that perform
    inference with a machine learning model. This class manages shared properties
    and handles model loading and device management via MLEngine.
    """

    __gstmetadata__ = (
        "GstBaseTransform",
        "Transform",
        "Generic machine learning model transform element",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    batch_size = GObject.Property(
        type=int,
        default=1,
        minimum=1,
        maximum=32,
        nick="Batch Size",
        blurb="Number of items to process in a batch",
        flags=GObject.ParamFlags.READWRITE,
    )

    frame_stride = GObject.Property(
        type=int,
        default=1,
        minimum=1,
        maximum=256,
        nick="Frame Stride",
        blurb="How often to process a frame",
        flags=GObject.ParamFlags.READWRITE,
    )
    device = GObject.Property(
        type=str,
        default="cpu",
        nick="Device",
        blurb="Device to run the inference on (cpu, cuda, cuda:0, cuda:1, etc.)",
        flags=GObject.ParamFlags.READWRITE,
    )

    model_name = GObject.Property(
        type=str,
        default=None,
        nick="Model Name",
        blurb="Name of the pre-trained model to load",
        flags=GObject.ParamFlags.READWRITE,
    )

    ml_engine = GObject.Property(
        type=str,
        default=None,
        nick="ML Engine",
        blurb="Machine Learning Engine to use : pytorch, tflite, tensorflow, onnx or openvino",
        flags=GObject.ParamFlags.READWRITE,
    )

    device_queue_id = GObject.Property(
        type=int,
        default=0,  # Default to queue ID 0
        minimum=0,
        maximum=32,  # You can adjust the maximum depending on the size of your pool
        nick="Device Queue ID",
        blurb="ID of the DeviceQueue from the pool to use",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        self.logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)
        self.ml_engine = GstEngineFactory.PYTORCH_ENGINE
        self.engine = None
        self.kwargs = {}

    def do_get_property(self, prop: GObject.ParamSpec):
        if prop.name == "batch-size":
            return self.batch_size
        elif prop.name == "frame-stride":
            return self.frame_stride
        elif prop.name == "model-name":
            return self.model_name
        elif prop.name == "device":
            if self.engine:
                return self.engine.get_device()
            return None
        elif prop.name == "ml-engine":
            return self.ml_engine
        elif prop.name == "device-queue-id":
            return self.device_queue_id
        else:
            raise AttributeError(f"Unknown property {prop.name}")

    def do_set_property(self, prop: GObject.ParamSpec, value):
        if prop.name == "batch-size":
            self.batch_size = value
            if self.engine:
                self.engine.batch_size = value
        elif prop.name == "frame-stride":
            self.frame_stride = value
            if self.engine:
                self.engine.frame_stride = value
        elif prop.name == "model-name":
            self.model_name = value
            self.do_load_model()
        elif prop.name == "device":
            self.device = value
            # Only set the device if the engine is initialized
            if self.engine:
                self.engine.set_device(value)
                self.do_load_model()
        elif prop.name == "ml-engine":
            if self.device:
                self.ml_engine = GstEngineFactory.create_engine(value, self.device)
                self.initialize_engine()
                self.do_load_model()
        elif prop.name == "device-queue-id":
            self.device_queue_id = value
            if self.engine:
                self.engine.device_queue_id = value
        else:
            raise AttributeError(f"Unknown property {prop.name}")

    def _initialize_engine_if_needed(self):
        """Initialize the engine if it hasn't been initialized yet."""
        if not self.engine and self.ml_engine:
            self.initialize_engine()

    def initialize_engine(self):
        """Initialize the machine learning engine based on the ml_engine property."""
        if self.ml_engine is not None:
            self.engine = GstEngineFactory.create_engine(self.ml_engine, self.device)
            self.engine.batch_size = self.batch_size
            self.engine.frame_stride = self.frame_stride
            if self.device_queue_id:
                self.engine.device_queue_id = self.device_queue_id
            self.do_load_model()
        else:
            Gst.error(f"Unsupported ML engine: {self.ml_engine}")
            return

    def do_load_model(self):
        """Loads the model using the current engine."""
        if self.engine and self.model_name:
            self.engine.load_model(self.model_name, **self.kwargs)
        else:
            Gst.warning("Engine is not present, unable to load the model.")

    def get_model(self):
        """Gets the model from the engine."""
        self._initialize_engine_if_needed()
        """Gets the model from the engine."""
        if self.engine:
            return self.engine.get_model()
        return None

    def set_model(self, model):
        """Sets the model in the engine."""
        if self.engine:
            self.engine.set_model(model)  # Set the model in the engine
            Gst.info("Model set successfully in the engine.")
        else:
            Gst.Error("Engine is not initialized.")
