# BaseAggregator (GStreamer backend)
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

"""GStreamer backend for the `aggregator` element family (input format differs
from output format, e.g. audio in / text out).

GStreamer half of the backend split: the element base (`GstBase.Aggregator`),
the GObject property declarations, and the framework virtuals
(`do_change_state`, `do_aggregate`, segment handling). Engine/model logic lives
in the portable `MLEngineMixin`.
"""

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
gi.require_version("GLib", "2.0")
from gi.repository import Gst, GObject, GstBase  # noqa: E402

from backend.core import MLEngineMixin, PayloadProcessingMixin  # noqa: E402


class PayloadDriver:
    """The GStreamer half of the payload seam.

    Kept apart from the element base so it can be driven without standing up a
    `GstBase.Aggregator`, the same way `FrameProcessingMixin` keeps the video
    work apart from the element that hosts it.
    """

    def do_process(self, buf):
        """Read the input payload, run the element's `process_payload`, and send
        each payload it returns as its own buffer, timed like the input.
        Elements supply `process_payload`, not this."""
        try:
            success, map_info = buf.map(Gst.MapFlags.READ)
            if not success:
                self.logger.error("Failed to map input buffer")
                return Gst.FlowReturn.ERROR

            payload = bytes(map_info.data)
            buf.unmap(map_info)

            for output in self.process_payload(payload):
                outbuf = Gst.Buffer.new_allocate(None, len(output), None)
                outbuf.fill(0, output)
                outbuf.pts = buf.pts
                outbuf.dts = buf.dts
                outbuf.duration = buf.duration
                self.push_payload(outbuf)

            return Gst.FlowReturn.OK

        except Exception as e:
            self.logger.error(f"Error processing buffer: {e}")
            return Gst.FlowReturn.ERROR

    def push_payload(self, outbuf):
        """Send one output buffer downstream.

        Through the aggregator by default. An element that has always pushed
        straight out of the src pad overrides this to keep doing that.
        """
        self.finish_buffer(outbuf)


class BaseAggregator(
    GstBase.Aggregator, MLEngineMixin, PayloadProcessingMixin, PayloadDriver
):
    """
    Base class for GStreamer aggregator elements that perform inference
    with a machine learning model. This class manages shared properties
    and handles model loading and device management via MLEngine.
    """

    __gstmetadata__ = (
        "BaseAggregator",
        "Aggregator",
        "Generic machine learning model aggregator element",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    def __init__(self):
        super().__init__()
        self._ml_init()
        self.segment_pushed = False

    @GObject.Property(type=str)
    def device(self):
        "Device to run the inference on (cpu, cuda, cuda:0, cuda:1, etc.)"
        return self.mgr.device

    @device.setter
    def device(self, value):
        self.mgr.set_device(value)
        # todo why is this needed ?
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
        "Machine Learning Engine to use : pytorch, tflite, tensorflow, onnx or openvino, or custom engine name"
        return self.mgr.engine_name

    @engine_name.setter
    def engine_name(self, value):
        self.mgr.engine_name = value

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

    # GStreamer framework virtual: load the model on NULL -> READY.
    def do_change_state(self, transition):
        if transition == Gst.StateChange.NULL_TO_READY:
            self.do_load_model()
        return Gst.Element.do_change_state(self, transition)

    def push_segment_if_needed(self):
        if not self.segment_pushed:
            segment = Gst.Segment()
            segment.init(Gst.Format.TIME)
            segment.start = 0
            segment.stop = Gst.CLOCK_TIME_NONE
            segment.position = 0

            self.srcpad.push_event(Gst.Event.new_segment(segment))
            self.segment_pushed = True

    # GStreamer framework virtual: pull buffers from sink pads and process.
    def do_aggregate(self, timeout):
        if all(pad.is_eos() for pad in self.sinkpads):
            return Gst.FlowReturn.EOS
        self.push_segment_if_needed()
        self.process_all_sink_pads()
        self.selected_samples(Gst.CLOCK_TIME_NONE, 0, 0, None)
        return Gst.FlowReturn.OK

    def process_all_sink_pads(self):
        if len(self.sinkpads) == 0:
            return
        buf = self.sinkpads[0].pop_buffer()
        if buf:
            self.do_process(buf)
