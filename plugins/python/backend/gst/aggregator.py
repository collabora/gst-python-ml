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

GStreamer half of the backend split: the element base (`GstBase.Aggregator`)
and the framework virtuals (`do_change_state`, `do_aggregate`, segment
handling). Engine/model logic lives in the portable `MLEngineMixin`, and the
shared tunables come from `ml_property_namespace`.
"""

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
gi.require_version("GLib", "2.0")
from gi.repository import Gst, GObject, GstBase  # noqa: E402

from backend.core import (  # noqa: E402
    MLEngineMixin,
    PayloadProcessingMixin,
    ml_property_namespace,
)


class PayloadDriver:
    """The GStreamer half of the payload seam.

    Kept apart from the element base so it can be driven without standing up a
    `GstBase.Aggregator`, the same way `FrameProcessingMixin` keeps the video
    work apart from the element that hosts it.
    """

    #: Send output straight out of the element's own src pad instead of through
    #: the aggregator. Two families have always done that.
    PUSH_FROM_SRC_PAD = False

    def do_process(self, buf):
        """Read the input payload, run the element's `process_payload`, and send
        each payload it returns as its own buffer. Elements supply
        `process_payload`, not this."""
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
                self.stamp_payload(outbuf, buf)
                self.push_payload(outbuf)

            return Gst.FlowReturn.OK

        except Exception as e:
            self.logger.error(f"Error processing buffer: {e}")
            return Gst.FlowReturn.ERROR

    def stamp_payload(self, outbuf, inbuf):
        """Time one output buffer, the gst spelling of what `meta.emit` takes.

        An element that generates media of its own length says so through
        `payload_duration_ns`; that audio runs for as long as it runs and plays
        wherever the pipeline reaches it, so the input's times say nothing about
        it. Everything else covers the same stretch of stream as its input.
        """
        duration_ns = self.payload_duration_ns(outbuf.get_size())
        if duration_ns is None:
            outbuf.pts = inbuf.pts
            outbuf.dts = inbuf.dts
            outbuf.duration = inbuf.duration
            return
        outbuf.pts = Gst.CLOCK_TIME_NONE
        outbuf.dts = Gst.CLOCK_TIME_NONE
        outbuf.duration = duration_ns

    def push_payload(self, outbuf):
        """Send one output buffer downstream."""
        if not self.PUSH_FROM_SRC_PAD:
            self.finish_buffer(outbuf)
            return
        ret = self.srcpad.push(outbuf)
        if ret != Gst.FlowReturn.OK:
            raise RuntimeError(f"Error pushing payload to pipeline: {ret}")


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

    # unpacked here rather than inherited: pygobject installs a property only
    # when it sits in the class's own dict
    locals().update(ml_property_namespace(GObject))

    def __init__(self):
        super().__init__()
        self._ml_init()
        self.segment_pushed = False

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
