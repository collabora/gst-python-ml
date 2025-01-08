# GstAggregator
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

from abc import abstractmethod
import gi
from engine.gst_engine_factory import GstEngineFactory

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
gi.require_version("GLib", "2.0")
from gi.repository import Gst, GObject, GstBase  # noqa: E402


class GstAggregator(GstBase.Aggregator):
    """
    Base class for GStreamer aggregator elements that perform inference
    with a machine learning model. This class manages shared properties
    and handles model loading and device management via MLEngine.
    """

    __gstmetadata__ = (
        "GstAggregator",
        "Aggregator",
        "Generic machine learning model aggregator element",
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
        blurb="Name of the pre-trained model or local model path",
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
        super(GstAggregator, self).__init__()
        self.ml_engine = GstEngineFactory.PYTORCH_ENGINE
        self.engine = None
        self.kwargs = {}
        self.segment_pushed = False

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
            else:
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
        else:
            Gst.warning("Engine is not present, unable to get the model.")
            return None

    def set_model(self, model):
        """Gets the model from the engine."""
        self._initialize_engine_if_needed()
        """Sets the model in the engine."""
        if self.engine:
            self.engine.set_model(model)
        else:
            Gst.warning("Engine is not present, unable to set the model.")

    def get_tokenizer(self):
        """Gets the model from the engine."""
        self._initialize_engine_if_needed()
        if self.get_model() is None:
            self.do_load_model()
        """Gets the model from the engine."""
        if self.engine:
            return self.engine.tokenizer
        else:
            Gst.warning("Engine is not present, unable to get the tokenizer.")
            return None

    def push_segment_if_needed(self):
        if not self.segment_pushed:
            segment = Gst.Segment()
            segment.init(Gst.Format.TIME)
            segment.start = 0
            segment.stop = Gst.CLOCK_TIME_NONE
            segment.position = 0

            self.srcpad.push_event(Gst.Event.new_segment(segment))
            self.segment_pushed = True

    def do_aggregate(self, timeout):
        """
        Aggregates the buffers from the sink pads,
        processes with the model, and pushes the result downstream.
        """
        self.push_segment_if_needed()
        self.process_all_sink_pads()
        return Gst.FlowReturn.OK

    def process_all_sink_pads(self):
        if len(self.sinkpads) == 0:
            return
        buf = self.sinkpads[0].pop_buffer()
        if buf:
            self.do_process(buf)

    @abstractmethod
    def do_process(self, buf):
        """Process a buffer using the loaded model."""
        pass
