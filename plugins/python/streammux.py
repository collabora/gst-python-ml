# StreamMux
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

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
gi.require_version("GObject", "2.0")
from gi.repository import Gst, GObject, GstBase  # noqa: E402

from log.logger_factory import LoggerFactory  # noqa: E402
from metadata import Metadata  # noqa: E402


class StreamMux(GstBase.Aggregator):
    __gstmetadata__ = (
        "StreamMux",
        "Video/Mux",
        "Custom stream muxer",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    __gsttemplates__ = (
        Gst.PadTemplate.new_with_gtype(
            "sink_%u",
            Gst.PadDirection.SINK,
            Gst.PadPresence.REQUEST,
            Gst.Caps.from_string("video/x-raw"),
            GstBase.AggregatorPad.__gtype__,
        ),
        Gst.PadTemplate.new_with_gtype(
            "src",
            Gst.PadDirection.SRC,
            Gst.PadPresence.ALWAYS,
            Gst.Caps.from_string("video/x-raw"),
            GstBase.AggregatorPad.__gtype__,
        ),
    )

    timeout = GObject.Property(
        type=int,
        default=5000,
        nick="Timeout",
        blurb="Timeout for batch aggregation (in milliseconds)",
    )

    def __init__(self):
        super().__init__()
        self.logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)
        self.batch_buffer = []
        self.timestamps = []
        self.timeout_source = None
        self.batch_size = 1  # Default batch size, dynamically adjusted
        self.metadata = Metadata("si")  # Use "si" for string ID and num_sources
        self.start_timeout()

    def start_timeout(self):
        """Start timeout for batch processing if not already running."""
        if self.timeout_source:
            return  # Already running
        self.timeout_source = GObject.timeout_add(self.timeout, self.handle_timeout)

    def stop_timeout(self):
        """Stop the timeout if it is running."""
        if self.timeout_source:
            GObject.source_remove(self.timeout_source)
            self.timeout_source = None

    def do_request_new_pad(self, templ, name, caps):
        """Handles requests for new sink pads."""
        self.logger.info(f"Requesting new sink pad: {name}")
        pad = Gst.Pad.new_from_template(templ, name)

        if not pad:
            self.logger.error(f"Failed to create pad {name}")
            return None

        self.add_pad(pad)
        return pad

    def handle_timeout(self):
        """Handle timeout event: process batch if not full yet."""
        if len(self.batch_buffer) > 0:
            self.output_batch()
        return True  # Keep the timeout active

    def do_aggregate(self, timeout):
        """Aggregates frames from all sink pads into a single batch."""
        self.batch_size = len(self.sinkpads)

        self.batch_buffer.clear()
        self.timestamps.clear()

        self.foreach_sink_pad(self.collect_frame, None)

        for pad in self.sinkpads:
            buf = pad.peek_buffer()
            if buf:
                structure = Gst.Structure.new_empty("selected-sample")
                self.selected_samples(buf.pts, buf.dts, buf.duration, structure)

        if len(self.batch_buffer) == self.batch_size:
            self.output_batch()

        return Gst.FlowReturn.OK

    def collect_frame(self, agg, pad, data):
        """Collect frames from all sink pads."""
        buf = pad.pop_buffer()
        if buf:
            self.batch_buffer.append(buf)
            self.timestamps.append(buf.pts)
        return True

    def output_batch(self):
        """Creates and sends a batched buffer downstream."""
        if len(self.batch_buffer) == 0 or len(self.timestamps) == 0:
            self.logger.warning("No buffers available, skipping batch output.")
            return

        num_sources = len(self.sinkpads)  # 🚀 Dynamically get active sink pads
        self.logger.info(f"Embedding num-sources={num_sources} into buffer memory")

        # Create a new buffer
        batch_buffer = Gst.Buffer.new()

        # Append video memory chunks FIRST
        for buf in self.batch_buffer:
            for i in range(buf.n_memory()):
                memory = buf.peek_memory(i)
                batch_buffer.append_memory(memory)

        # Append metadata LAST with string ID "mux/demux" and num_sources
        self.metadata.write(batch_buffer, "mux/demux", num_sources)

        # Log metadata memory
        with batch_buffer.peek_memory(batch_buffer.n_memory() - 1).map(
            Gst.MapFlags.READ
        ) as map_info:
            self.logger.info(f"Muxer - Created metadata memory: {map_info.data.hex()}")

        # 🚨 Log stream-start event
        if not hasattr(self, "stream_started"):
            self.logger.info("Sending STREAM-START event from StreamMux")
            self.srcpad.push_event(Gst.Event.new_stream_start("mux-stream"))
            self.stream_started = True

        # 🚨 Log caps negotiation
        first_pad = next(iter(self.sinkpads), None)
        if first_pad and first_pad.has_current_caps():
            in_caps = first_pad.get_current_caps()
            self.logger.info(f"Setting CAPS on StreamMux src pad: {in_caps}")
            self.srcpad.set_caps(in_caps)
        else:
            self.logger.error("No input caps available in StreamMux!")

        # 🚨 Log segment event
        segment = Gst.Segment()
        segment.init(Gst.Format.TIME)
        segment.start = min(self.timestamps) if self.timestamps else 0
        self.logger.info(f"Sending SEGMENT event with start={segment.start}")
        self.srcpad.push_event(Gst.Event.new_segment(segment))

        # 🚨 Log buffer push
        batch_buffer.pts = segment.start
        self.logger.debug("Pushing buffer from StreamMux")

        # Debug metadata
        with batch_buffer.peek_memory(batch_buffer.n_memory() - 1).map(
            Gst.MapFlags.READ
        ) as map_info:
            self.logger.debug(f"Buffer last memory before push: {map_info.data.hex()}")

        self.finish_buffer(batch_buffer)

    def do_sink_event(self, pad, event):
        """Handles sink pad events, including latency queries."""
        if event.type == Gst.EventType.LATENCY:
            self.logger.info("Received LATENCY event, updating pipeline latency.")
            self.aggregator_update_latency()
            return True  # Mark event as handled
        return GstBase.Aggregator.do_sink_event(self, pad, event)

    def do_set_property(self, prop, value):
        if prop.name == "timeout":
            self.timeout = value
            self.start_timeout()
        else:
            raise AttributeError(f"Unknown property: {prop.name}")

    def do_get_property(self, prop):
        if prop.name == "timeout":
            return self.timeout
        else:
            raise AttributeError(f"Unknown property: {prop.name}")


GObject.type_register(StreamMux)
__gstelementfactory__ = ("pyml_streammux", Gst.Rank.NONE, StreamMux)
