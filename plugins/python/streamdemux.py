# StreamDemux
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
gi.require_version("GLib", "2.0")

from gi.repository import Gst, GObject  # noqa: E402


class StreamDemux(Gst.Element):
    __gstmetadata__ = (
        "StreamDemux",
        "Demuxer",
        "Custom stream demuxer",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    __gsttemplates__ = (
        Gst.PadTemplate.new(
            "sink",
            Gst.PadDirection.SINK,
            Gst.PadPresence.ALWAYS,
            Gst.Caps.from_string("video/x-raw"),
        ),
        Gst.PadTemplate.new(
            "src_%u",
            Gst.PadDirection.SRC,
            Gst.PadPresence.REQUEST,
            Gst.Caps.from_string("video/x-raw"),
        ),
    )

    def __init__(self):
        super(StreamDemux, self).__init__()
        self.sinkpad = Gst.Pad.new_from_template(self.get_pad_template("sink"), "sink")
        self.sinkpad.set_event_function_full(self.event)
        self.sinkpad.set_chain_function_full(self.chain)
        self.add_pad(self.sinkpad)
        self.pad_count = 0  # Keep track of dynamic pads

    def do_request_new_pad(self, template, name, caps):
        # If no name is provided, generate a unique one
        if name is None:
            name = f"src_{self.pad_count}"  # Increment pad count for unique name
            self.pad_count += 1

        Gst.debug(f"Requesting new pad: {name}")

        # Create and add the dynamic src pad
        if "src_" in name:
            pad = Gst.Pad.new_from_template(template, name)
            self.add_pad(pad)  # Add the dynamically created pad to the element
            return pad

        return None

    def do_release_pad(self, pad):
        pad_name = pad.get_name()
        Gst.debug(f"Releasing pad: {pad_name}")
        self.remove_pad(pad)  # Remove the dynamic pad

    def process_src_pad(self, pad, src_pad, buffer, memory_chunk):
        """Push memory chunk to the src pad."""
        out_buffer = Gst.Buffer.new()  # Create a new buffer for the memory chunk
        out_buffer.append_memory(memory_chunk)  # Add the memory chunk to the buffer

        # Copy buffer's timestamp and other relevant metadata
        out_buffer.pts = buffer.pts
        out_buffer.duration = buffer.duration
        out_buffer.dts = buffer.dts
        out_buffer.offset = buffer.offset

        # Push the buffer to the src pad
        ret = src_pad.push(out_buffer)
        if ret != Gst.FlowReturn.OK:
            Gst.error(f"Failed to push buffer on {src_pad.get_name()}: {ret}")

    def chain(self, pad, parent, buffer):
        Gst.debug("Processing buffer in chain function")

        # Get the number of memory chunks in the incoming buffer
        num_memory_chunks = buffer.n_memory()

        # Iterate over the memory chunks
        for idx in range(num_memory_chunks):
            memory_chunk = buffer.peek_memory(idx)  # Get the memory chunk

            # Request or find the corresponding src pad
            pad_name = f"src_{idx}"
            src_pad = self.get_static_pad(pad_name)

            if src_pad is None:
                src_pad = self.request_pad(
                    self.get_pad_template("src_%u"), pad_name, None
                )
                if src_pad is None:
                    Gst.error(f"Failed to request or create pad: {pad_name}")
                    continue

            # Push the memory chunk to the corresponding src pad
            self.process_src_pad(pad, src_pad, buffer, memory_chunk)

        return Gst.FlowReturn.OK

    def event(self, pad, parent, event):
        Gst.debug(f"Received event: {event.type}")
        return Gst.PadProbeReturn.OK


GObject.type_register(StreamDemux)
__gstelementfactory__ = ("pyml_streamdemux", Gst.Rank.NONE, StreamDemux)
