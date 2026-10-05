# InPlaceVideoTransform (GStreamer backend)
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


import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import Gst, GstBase, GstVideo  # noqa: E402

from backend.gst.errors import post_error  # noqa: E402
from log.logger_factory import LoggerFactory  # noqa: E402


class InPlaceVideoTransform(GstBase.BaseTransform):
    VIDEO_CAPS = Gst.Caps.from_string("video/x-raw")
    __gsttemplates__ = (
        Gst.PadTemplate.new(
            "src", Gst.PadDirection.SRC, Gst.PadPresence.ALWAYS, VIDEO_CAPS
        ),
        Gst.PadTemplate.new(
            "sink", Gst.PadDirection.SINK, Gst.PadPresence.ALWAYS, VIDEO_CAPS
        ),
    )

    # an element that only reads and writes metadata leaves this False
    READS_PIXELS = True

    def __init__(self):
        super().__init__()
        self.logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)
        self.set_in_place(True)
        self.set_passthrough(not self.READS_PIXELS)
        self.width = 0
        self.height = 0
        self.format = None
        self._bytes_per_pixel = 0

    def do_set_caps(self, incaps, outcaps):
        info = GstVideo.VideoInfo.new_from_caps(incaps)
        self.width = info.width
        self.height = info.height
        self.format = info.finfo.name
        self._bytes_per_pixel = info.finfo.pixel_stride[0]
        return True

    def do_transform_ip(self, buf):
        import numpy as np

        if not self.READS_PIXELS:
            return self._run(None, buf)
        ok, mapinfo = buf.map(Gst.MapFlags.WRITE)
        if not ok:
            self.logger.error("failed to map buffer for writing")
            return Gst.FlowReturn.ERROR
        try:
            count = self.width * self.height * self._bytes_per_pixel
            frame = np.frombuffer(mapinfo.data, dtype=np.uint8, count=count)
            return self._run(
                frame.reshape(self.height, self.width, self._bytes_per_pixel), buf
            )
        finally:
            buf.unmap(mapinfo)

    def _run(self, frame, buf):
        try:
            self.process_in_place(frame, self.format, buf)
            return Gst.FlowReturn.OK
        except Exception as exception:
            post_error(self, "transform error", exception)
            return Gst.FlowReturn.ERROR
