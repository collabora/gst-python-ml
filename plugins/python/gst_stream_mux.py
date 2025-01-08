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
        blurb="Timeout for batch aggregation (in miliseconds)",
    )

    def __init__(self):
        super(StreamMux, self).__init__()
        self.batch_buffer = []
        self.timestamps = []
        self.timeout_source = None
        self.start_timeout()

    def start_timeout(self):
        if self.timeout_source:
            self.timeout_source.destroy()

        self.timeout_source = GObject.timeout_add(self.timeout, self.handle_timeout)

    def stop_timeout(self):
        if self.timeout_source:
            GObject.source_remove(self.timeout_source)
            self.timeout_source = None

    def handle_timeout(self):
        if len(self.batch_buffer) > 0:
            self.output_batch()
        return True  # Keep the timeout active

    def do_aggregate(self, timeout):
        """
        Aggregates frames from all sink pads into a single batch.
        """
        # Dynamically set batch size based on the number of active sink pads
        self.batch_size = len(self.sinkpads)

        # Clear the previous batch
        self.batch_buffer.clear()
        self.timestamps.clear()

        # Collect frames from all sink pads
        self.foreach_sink_pad(self.collect_frame, None)

        # If the batch is full, output it downstream
        if len(self.batch_buffer) == self.batch_size:
            self.output_batch()

        return Gst.FlowReturn.OK

    def collect_frame(self, agg, pad, data):
        buf = pad.pop_buffer()
        if buf:
            self.batch_buffer.append(buf)
            self.timestamps.append(buf.pts)

        return True

    def output_batch(self):
        if len(self.batch_buffer) == 0:
            return

        # Create a new buffer to store the batched frames
        batch_buffer = Gst.Buffer.new()

        # Add each sub-buffer as separate GstMemory to the batched buffer
        for buf in self.batch_buffer:
            memory = buf.peek_memory(0)
            batch_buffer.append_memory(memory)

        # Set the PTS of the batched buffer to the minimum timestamp from the batch
        batch_buffer.pts = min(self.timestamps)

        # Send the batched buffer downstream
        self.finish_buffer(batch_buffer)

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
__gstelementfactory__ = ("streammux_pyml", Gst.Rank.NONE, StreamMux)
