# VideoTransform (GStreamer backend)
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

import time

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import Gst, GstBase  # noqa: E402

from backend.core import FrameProcessingMixin  # noqa: E402
from backend.gst.errors import post_error  # noqa: E402
from backend.gst.transform import BaseTransform  # noqa: E402

# GST_BASE_TRANSFORM_FLOW_DROPPED, which pygobject does not expose
FLOW_DROPPED = Gst.FlowReturn.CUSTOM_SUCCESS


class VideoTransform(BaseTransform, FrameProcessingMixin):
    """
    GStreamer element for video transformation using a PyTorch model.
    """

    # Define VIDEO_CAPS to support multiple formats
    VIDEO_CAPS = Gst.Caps.from_string(
        "video/x-raw,format=(string){ RGB, RGBA, ARGB, BGRA, ABGR },"
        "width=(int)[1,2147483647],height=(int)[1,2147483647]"
    )
    __gsttemplates__ = (
        Gst.PadTemplate.new(
            "src", Gst.PadDirection.SRC, Gst.PadPresence.ALWAYS, VIDEO_CAPS
        ),
        Gst.PadTemplate.new(
            "sink", Gst.PadDirection.SINK, Gst.PadPresence.ALWAYS, VIDEO_CAPS
        ),
    )

    def __init__(self):
        super().__init__()
        self.set_qos_enabled(True)
        self._sink_running_time_ns = None
        self._sink_clock_then_ns = None
        self._last_run_ns = 0

    def do_src_event(self, event):
        if event.type == Gst.EventType.QOS:
            _, _, diff, timestamp = event.parse_qos()
            clock = self.get_clock()
            if diff > 0 and clock is not None:
                self._sink_running_time_ns = timestamp + diff
                self._sink_clock_then_ns = clock.get_time()
        return GstBase.BaseTransform.do_src_event(self, event)

    def do_sink_event(self, event):
        if event.type == Gst.EventType.FLUSH_STOP:
            self._sink_running_time_ns = None
        return GstBase.BaseTransform.do_sink_event(self, event)

    def would_reach_the_sink_late(self, buf):
        if self._sink_running_time_ns is None or buf.pts == Gst.CLOCK_TIME_NONE:
            return False
        running_time = self.segment.to_running_time(Gst.Format.TIME, buf.pts)
        if running_time == Gst.CLOCK_TIME_NONE:
            return False
        elapsed_ns = self.get_clock().get_time() - self._sink_clock_then_ns
        due_by = self._sink_running_time_ns + elapsed_ns + self._last_run_ns
        return running_time < due_by

    def do_set_caps(self, incaps, outcaps):
        struct = incaps.get_structure(0)
        self.width = struct.get_int("width").value
        self.height = struct.get_int("height").value

        return True

    def do_transform_ip(self, buf):
        """GStreamer per-frame driver: extract the frame(s) through the backend
        frame I/O, run the element's `process_frames`, and map the outcome to a
        `Gst.FlowReturn`. Elements supply `process_frames`, not this."""
        # Imported lazily: the frameio singleton lives in backend.gst, which is
        # still being constructed when this module is imported.
        from backend import frameio

        if self._only_on and not self.carries_only_on(buf):
            return Gst.FlowReturn.OK
        if self.is_qos_enabled() and self.would_reach_the_sink_late(buf):
            return FLOW_DROPPED
        try:
            frames, num_sources, fmt = frameio.read_frames(
                buf,
                self.sinkpad,
                self.width,
                self.height,
                (
                    getattr(self, "framerate_num", 30),
                    getattr(self, "framerate_denom", 1),
                ),
            )
            if frames is None:
                self.logger.error("Failed to extract frames")
                return Gst.FlowReturn.ERROR
            started = time.monotonic_ns()
            self.process_frames(frames, num_sources, fmt, buf)
            self._last_run_ns = time.monotonic_ns() - started
            return Gst.FlowReturn.OK
        except Exception as exception:
            post_error(self, "transform error", exception)
            return Gst.FlowReturn.ERROR
