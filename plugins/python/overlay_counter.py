# OverlayCounter
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

from global_logger import GlobalLogger

CAN_REGISTER_ELEMENT = True
try:
    from gi.repository import Gst, GObject

    from overlay import Overlay
    from overlay_utils import Color
except ImportError as e:
    GlobalLogger().warning(
        f"The 'OverlayCounter' element cannot be registered because: {e}"
    )
    CAN_REGISTER_ELEMENT = False


class OverlayCounter(Overlay):

    __gstmetadata__ = (
        "OverlayCounter",
        "Filter/Effect/Video",
        "Counts objects and displays count",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    def do_set_dims(self, width, height):
        super().do_set_dims(width, height)
        self.tracking_display.set_y_line(self.height / 2)

    def do_post_process(self, frame_metadata):
        # Call the base class's method to display tracks, bounding boxes and labels
        super().do_post_process(frame_metadata)

        # add some graphics
        # Define the line start and end points
        start_point = {"x": 0, "y": self.tracking_display.y_line}
        end_point = {"x": self.width, "y": self.tracking_display.y_line}

        # Define the color (red with full opacity) and line width
        red_color = Color(r=1.0, g=0.0, b=0.0, a=1.0)  # RGBA
        line_width = 5.0

        # Draw the line
        self.overlay_graphics.draw_line(
            start=start_point, end=end_point, color=red_color, width=line_width
        )

        approaching, receding = self.tracking_display.count_objects()

        text = f"Objects receding : {receding}, approaching : {approaching}"
        self.overlay_graphics.draw_text(text, 0, 50, Color(1, 0, 0, 1), 20)


if CAN_REGISTER_ELEMENT:
    GObject.type_register(OverlayCounter)
    __gstelementfactory__ = ("pyml_overlay_counter", Gst.Rank.NONE, OverlayCounter)
else:
    GlobalLogger().warning("Failed to register the 'pyml_overlay_counter' element.")
