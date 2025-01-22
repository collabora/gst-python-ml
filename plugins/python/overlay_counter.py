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

CAN_REGISTER_ELEMENT = True
try:
    from overlay import Overlay
    from overlay_utils import Color
    from gi.repository import Gst, GObject
except ImportError as e:
    Gst.warning(f"The 'OverlayCounter' element cannot be registered because: {e}")
    CAN_REGISTER_ELEMENT = False


class OverlayCounter(Overlay):
    def do_post_process(self, frame_metadata):
        # Call the base class's method to display tracks, bounding boxes and labels
        super().do_post_process(frame_metadata)

        line_height = self.height / 2
        self.tracking_display.set_y_line(line_height)

        # add some graphics
        # Define the line start and end points
        start_point = {"x": 0, "y": line_height}
        end_point = {"x": self.width, "y": line_height}

        # Define the color (red with full opacity) and line width
        red_color = Color(r=1.0, g=0.0, b=0.0, a=1.0)  # RGBA
        line_width = 5.0

        # Draw the line
        self.overlay_graphics.draw_line(
            start=start_point, end=end_point, color=red_color, width=line_width
        )

        up, down = self.tracking_display.update_objects_crossing_line()

        text = f"Cars going up {up}, cars going down {down}"
        self.overlay_graphics.draw_text(text, 0, 50)


if CAN_REGISTER_ELEMENT:
    GObject.type_register(OverlayCounter)
    __gstelementfactory__ = ("pyml_overlay_counter", Gst.Rank.NONE, OverlayCounter)
else:
    Gst.warning("Failed to register the 'OverlayCounter' element.")
