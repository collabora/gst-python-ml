# VideoTransform
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
gi.require_version("GstVideo", "1.0")
from gi.repository import Gst, GObject  # noqa: E402

from transform_base import TransformBase  # noqa: E402


class VideoTransform(TransformBase):
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

    # Add properties using GObject.Property
    downsampled_width = GObject.Property(
        type=int,
        default=0,
        nick="Downsampled Width",
        blurb="The width of a downsampled replica of the output video",
        flags=GObject.ParamFlags.READWRITE,
    )

    downsampled_height = GObject.Property(
        type=int,
        default=0,
        nick="Downsampled Height",
        blurb="The height of a downsampled replica of the output video",
        flags=GObject.ParamFlags.READWRITE,
    )

    def do_set_property(self, prop, value):
        if prop.name == "downsampled-width":
            self.downsampled_width = value
        elif prop.name == "downsampled-height":
            self.downsampled_height = value
        else:
            super().do_set_property(prop, value)

    def do_get_property(self, prop):
        if prop.name == "downsampled-width":
            return self.downsampled_width
        elif prop.name == "downsampled-height":
            return self.downsampled_height
        return super().do_get_property(prop)

    def do_set_caps(self, incaps, outcaps):
        struct = incaps.get_structure(0)
        self.width = struct.get_int("width").value
        self.height = struct.get_int("height").value

        return True
