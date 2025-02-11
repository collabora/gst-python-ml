# AggregatorBase
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

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
gi.require_version("GLib", "2.0")
from gi.repository import Gst, GObject, GstBase  # noqa: E402

from engine.engine_factory import EngineFactory
from log.logger_factory import LoggerFactory
from model_engine_helper import ModelEngineHelper


class AggregatorBase(GstBase.Aggregator):
    """
    Base class for GStreamer aggregator elements that perform inference
    with a machine learning model. This class manages shared properties
    and handles model loading and device management via MLEngine.
    """

    __gstmetadata__ = (
        "AggregatorBase",
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
    engine_name = GObject.Property(
        type=str,
        default=None,
        nick="ML Engine",
        blurb="Machine Learning Engine to use : pytorch, tflite, tensorflow, onnx or openvino",
        flags=GObject.ParamFlags.READWRITE,
    )

    device_queue_id = GObject.Property(
        type=int,
        default=0,
        minimum=0,
        maximum=32,
        nick="Device Queue ID",
        blurb="ID of the DeviceQueue from the pool to use",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        self.logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)
        self.engine_helper = ModelEngineHelper(self.logger)
        self.engine_name = self.engine_helper.engine_name
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
            return self.device  # Return from AggregatorBase, not from helper
        elif prop.name == "engine-name":
            return self.engine_name
        elif prop.name == "device-queue-id":
            return self.device_queue_id
        else:
            raise AttributeError(f"Unknown property {prop.name}")

    def do_set_property(self, prop: GObject.ParamSpec, value):
        if prop.name == "batch-size":
            self.batch_size = value
            if self.engine_helper.engine:
                self.engine_helper.engine.batch_size = value
        elif prop.name == "frame-stride":
            self.frame_stride = value
            if self.engine_helper.engine:
                self.engine_helper.engine.frame_stride = value
        elif prop.name == "model-name":
            self.model_name = value
            self.engine_helper.load_model(value)
        elif prop.name == "device":
            self.device = value
            self.engine_helper.set_device(value)  # Update device in helper
            self.engine_helper.initialize_engine(self.engine_name)
            self.engine_helper.load_model(self.model_name)
        elif prop.name == "engine-name":
            self.engine_name = value
            self.engine_helper.initialize_engine(value)
            self.engine_helper.load_model(self.model_name)
        elif prop.name == "device-queue-id":
            self.device_queue_id = value
            if self.engine_helper.engine:
                self.engine_helper.engine.device_queue_id = value
        else:
            raise AttributeError(f"Unknown property {prop.name}")

    def _initialize_engine_if_needed(self):
        if not self.engine_helper.engine and self.engine_name:
            self.engine_helper.initialize_engine(self.engine_name)

    def initialize_engine(self):
        if self.engine_name is not None:
            self.engine_helper.initialize_engine(self.engine_name)
            self.engine_helper.engine.batch_size = self.batch_size
            self.engine_helper.engine.frame_stride = self.frame_stride
            if self.device_queue_id:
                self.engine_helper.engine.device_queue_id = self.device_queue_id
        else:
            self.logger.error(f"Unsupported ML engine: {self.engine_name}")

    def do_load_model(self):
        if self.engine_helper.engine and self.model_name:
            self.engine_helper.load_model(self.model_name)
        else:
            self.logger.warning("Engine is not present, unable to load the model.")

    def get_model(self):
        self._initialize_engine_if_needed()
        if self.engine_helper.engine:
            return self.engine_helper.get_model()
        else:
            self.logger.warning("Engine is not present, unable to get the model.")
            return None

    def set_model(self, model):
        self._initialize_engine_if_needed()
        if self.engine_helper.engine:
            self.engine_helper.set_model(model)
        else:
            self.logger.warning("Engine is not present, unable to set the model.")

    def get_tokenizer(self):
        self._initialize_engine_if_needed()
        if self.engine_helper.engine:
            return self.engine_helper.get_tokenizer()
        else:
            self.logger.warning("Engine is not present, unable to get the tokenizer.")
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
        pass
