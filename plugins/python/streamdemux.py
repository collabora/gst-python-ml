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

from log.logger_factory import LoggerFactory


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
        super().__init__()
        self.logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)
        self.sinkpad = Gst.Pad.new_from_template(self.get_pad_template("sink"), "sink")
        self.sinkpad.set_event_function_full(self.event)
        self.sinkpad.set_chain_function_full(self.chain)
        self.add_pad(self.sinkpad)
        self.pad_count = 0  # Keep track of dynamic pads

    def do_request_new_pad(self, template, name, caps):
        if name is None:
            name = f"src_{self.pad_count}"
            self.pad_count += 1

        self.logger.debug(f"Requesting new pad: {name}")

        if "src_" in name:
            pad = Gst.Pad.new_from_template(template, name)
            self.add_pad(pad)

            # Ensure stream-start event is pushed FIRST
            if not hasattr(pad, "stream_started"):
                pad.push_event(Gst.Event.new_stream_start(f"demux-stream-{name}"))
                pad.stream_started = True

            # Ensure caps are set from sinkpad BEFORE pushing any buffer
            if self.sinkpad.has_current_caps():
                caps = self.sinkpad.get_current_caps()
                self.logger.info(f"Setting caps on {pad.get_name()}: {caps}")
                pad.set_caps(caps)
            else:
                self.logger.warning(f"Cannot set caps on {pad.get_name()}, sinkpad has no caps!")

            return pad
        return None



    def do_release_pad(self, pad):
        pad_name = pad.get_name()
        self.logger.debug(f"Releasing pad: {pad_name}")
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
            self.logger.error(f"Failed to push buffer on {src_pad.get_name()}: {ret}")

    def chain(self, pad, parent, buffer):
        self.logger.debug("Processing buffer in chain function")

        num_memory_chunks = buffer.n_memory()

        for idx in range(num_memory_chunks):
            memory_chunk = buffer.peek_memory(idx)

            pad_name = f"src_{idx}"
            src_pad = self.get_static_pad(pad_name)

            if src_pad is None:
                src_pad = self.request_pad(
                    self.get_pad_template("src_%u"), pad_name, None
                )
                if src_pad is None:
                    self.logger.error(f"Failed to request or create pad: {pad_name}")
                    continue

            # 🚨 Ensure stream-start event
            if not hasattr(src_pad, "stream_started"):
                self.logger.info(f"Sending STREAM-START event on {src_pad.get_name()}")
                src_pad.push_event(Gst.Event.new_stream_start(f"demux-stream-{idx}"))
                src_pad.stream_started = True

            # 🚨 Ensure caps are set
            if not src_pad.has_current_caps():
                if self.sinkpad.has_current_caps():
                    caps = self.sinkpad.get_current_caps()
                    self.logger.info(f"Setting CAPS on {src_pad.get_name()}: {caps}")
                    src_pad.set_caps(caps)
                else:
                    self.logger.error(f"No CAPS found on sinkpad. Cannot push buffer.")
                    return Gst.FlowReturn.NOT_NEGOTIATED

            # 🚨 Ensure segment event
            if not hasattr(src_pad, "segment_pushed"):
                segment = Gst.Segment()
                segment.init(Gst.Format.TIME)
                segment.start = buffer.pts
                self.logger.info(f"Sending SEGMENT event on {src_pad.get_name()} with start={segment.start}")
                src_pad.push_event(Gst.Event.new_segment(segment))
                src_pad.segment_pushed = True

            # 🚨 Log buffer push
            self.logger.info(f"Pushing buffer on {src_pad.get_name()}")
            self.process_src_pad(pad, src_pad, buffer, memory_chunk)

        return Gst.FlowReturn.OK


    def event(self, pad, parent, event):
        self.logger.debug(f"Received event: {event.type}")
        return Gst.PadProbeReturn.OK


GObject.type_register(StreamDemux)
__gstelementfactory__ = ("pyml_streamdemux", Gst.Rank.NONE, StreamDemux)
