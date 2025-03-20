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
gi.require_version("GstAnalytics", "1.0")
gi.require_version("GLib", "2.0")
from gi.repository import Gst, GObject, GstAnalytics, GLib  # noqa: E402
from log.logger_factory import LoggerFactory  # noqa: E402
from metadata import Metadata  # noqa: E402
from collections import defaultdict  # noqa: E402


class StreamDemux(Gst.Element):
    __gstmetadata__ = (
        "StreamDemux",
        "Demuxer",
        "Custom stream demuxer with metadata splitting",
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

    __gproperties__ = {
        "max-queue-size": (
            GObject.TYPE_UINT,
            "Maximum queue size",
            "Maximum number of buffers to queue per source pad",
            1,
            1000,
            10,
            GObject.ParamFlags.READWRITE,
        ),
    }

    def __init__(self):
        super().__init__()
        self.logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)
        self.sinkpad = Gst.Pad.new_from_template(self.get_pad_template("sink"), "sink")
        self.sinkpad.set_event_function_full(self.event)
        self.sinkpad.set_chain_function_full(self.chain)
        self.add_pad(self.sinkpad)
        self.pad_count = 0
        self.metadata = Metadata("si")
        self.buffer_queues = defaultdict(list)
        self.src_pads = {}
        self.max_queue_size = 10
        self.pads_requested = False  # Track initial pad creation

    def do_set_property(self, prop, value):
        if prop.name == "max-queue-size":
            self.max_queue_size = value
        else:
            raise AttributeError(f"Unknown property: {prop.name}")

    def do_get_property(self, prop):
        if prop.name == "max-queue-size":
            return self.max_queue_size
        raise AttributeError(f"Unknown property: {prop.name}")

    def do_request_new_pad(self, template, name, caps):
        if name is None:
            name = f"src_{self.pad_count}"
            self.pad_count += 1
        self.logger.debug(f"Requesting new pad: {name}")
        if "src_" in name:
            pad = Gst.Pad.new_from_template(template, name)
            self.add_pad(pad)
            if not hasattr(pad, "stream_started"):
                pad.push_event(Gst.Event.new_stream_start(f"demux-stream-{name}"))
                pad.stream_started = True
            if self.sinkpad.has_current_caps():
                caps = self.sinkpad.get_current_caps()
                self.logger.info(f"Setting caps on {pad.get_name()}: {caps}")
                pad.set_caps(caps)
            self.src_pads[name] = pad
            return pad
        return None

    def do_release_pad(self, pad):
        pad_name = pad.get_name()
        self.logger.debug(f"Releasing pad: {pad_name}")
        self.buffer_queues[pad_name].clear()
        self.remove_pad(pad)
        del self.src_pads[pad_name]

    def process_src_pad(self, buffer, memory_chunk, stream_idx):
        out_buffer = Gst.Buffer.new()
        out_buffer.append_memory(memory_chunk)
        out_buffer.pts = buffer.pts
        out_buffer.duration = buffer.duration
        out_buffer.dts = buffer.dts
        out_buffer.offset = buffer.offset
        meta = GstAnalytics.buffer_get_analytics_relation_meta(buffer)
        if meta:
            out_meta = GstAnalytics.buffer_add_analytics_relation_meta(out_buffer)
            count = GstAnalytics.relation_get_length(meta)
            self.logger.info(
                f"Processing {count} analytics relations for stream_{stream_idx}"
            )
            for i in range(count):
                ret, od_mtd = meta.get_od_mtd(i)
                if ret and od_mtd:
                    label_quark = od_mtd.get_obj_type()
                    label = GLib.quark_to_string(label_quark)
                    if f"stream_{stream_idx}_" in label:
                        presence, x, y, w, h, conf = od_mtd.get_location()
                        if presence:
                            qk = GLib.quark_from_string(label)
                            ret, new_od_mtd = out_meta.add_od_mtd(qk, x, y, w, h, conf)
                            if not ret:
                                self.logger.error(f"Failed to attach metadata: {label}")
        return out_buffer

    def push_buffer(self, pad, buffer, pad_name):
        self.logger.debug(f"Pushing buffer on {pad_name}")
        ret = pad.push(buffer)
        if ret != Gst.FlowReturn.OK:
            self.logger.error(f"Failed to push buffer on {pad_name}: {ret}")
            if ret not in (Gst.FlowReturn.FLUSHING, Gst.FlowReturn.ERROR):
                if len(self.buffer_queues[pad_name]) < self.max_queue_size:
                    self.buffer_queues[pad_name].append(buffer)
                else:
                    self.logger.warning(
                        f"Dropping buffer on {pad_name}: queue full (size={self.max_queue_size})"
                    )
            return False
        self.logger.debug(f"Successfully pushed buffer on {pad_name}")
        return True

    def chain(self, pad, parent, buffer):
        self.logger.debug("Processing buffer in chain function")
        if buffer.n_memory() > 0:
            try:
                id_str, num_sources = self.metadata.read(buffer)
                self.logger.info(f"Decoded ID: {id_str}, num_sources: {num_sources}")
                # Request all pads upfront on first buffer
                if not self.pads_requested:
                    for idx in range(num_sources):
                        pad_name = f"src_{idx}"
                        if pad_name not in self.src_pads:
                            self.request_pad(
                                self.get_pad_template("src_%u"), pad_name, None
                            )
                    self.pads_requested = True
                    # Ensure caps are set on all pads
                    if self.sinkpad.has_current_caps():
                        caps = self.sinkpad.get_current_caps()
                        for src_pad in self.src_pads.values():
                            if not src_pad.has_current_caps():
                                self.logger.info(
                                    f"Setting caps on {src_pad.get_name()}: {caps}"
                                )
                                src_pad.set_caps(caps)
            except ValueError as e:
                self.logger.error(str(e))

        num_memory_chunks = buffer.n_memory() - 1
        for idx in range(num_memory_chunks):
            memory_chunk = buffer.peek_memory(idx)
            pad_name = f"src_{idx}"
            src_pad = self.get_static_pad(pad_name)
            if src_pad is None:
                src_pad = self.request_pad(
                    self.get_pad_template("src_%u"), pad_name, None
                )
                if src_pad is None:
                    self.logger.error(f"Failed to request pad: {pad_name}")
                    continue
            if not hasattr(src_pad, "stream_started"):
                src_pad.push_event(Gst.Event.new_stream_start(f"demux-stream-{idx}"))
                src_pad.stream_started = True
            if not src_pad.has_current_caps() and self.sinkpad.has_current_caps():
                caps = self.sinkpad.get_current_caps()
                self.logger.info(f"Setting CAPS on {src_pad.get_name()}: {caps}")
                src_pad.set_caps(caps)
            if not hasattr(src_pad, "segment_pushed"):
                segment = Gst.Segment()
                segment.init(Gst.Format.TIME)
                segment.start = buffer.pts
                self.logger.info(
                    f"Sending SEGMENT event on {src_pad.get_name()} with start={segment.start}"
                )
                src_pad.push_event(Gst.Event.new_segment(segment))
                src_pad.segment_pushed = True

            out_buffer = self.process_src_pad(buffer, memory_chunk, stream_idx=idx)
            while self.buffer_queues[pad_name]:
                if self.push_buffer(
                    src_pad, self.buffer_queues[pad_name].pop(0), pad_name
                ):
                    continue
                break
            self.push_buffer(src_pad, out_buffer, pad_name)
        return Gst.FlowReturn.OK

    def event(self, pad, parent, event):
        self.logger.debug(f"Received event: {event.type}")
        return Gst.PadProbeReturn.OK


GObject.type_register(StreamDemux)
__gstelementfactory__ = ("pyml_streamdemux", Gst.Rank.NONE, StreamDemux)
