# overlay_utils_cairo.py
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

import cairo
from .overlay_utils_interface import OverlayGraphics, Color


class CairoOverlayGraphics(OverlayGraphics):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.surface = None
        self.context = None

    def initialize(self, buffer_data):
        """Initialize or reuse Cairo surface and context for drawing."""
        if (
            self.surface is None
            or self.surface.get_width() != self.width
            or self.surface.get_height() != self.height
        ):
            # Create a new surface if it doesn't exist or dimensions have changed
            self.surface = cairo.ImageSurface.create_for_data(
                buffer_data,
                cairo.FORMAT_ARGB32,
                self.width,
                self.height,
                self.width * 4,
            )
            self.context = cairo.Context(self.surface)
        else:
            # Reuse existing surface and update its data
            self.surface.flush()  # Ensure all operations on the surface are complete
            self.surface = cairo.ImageSurface.create_for_data(
                buffer_data,
                cairo.FORMAT_ARGB32,
                self.width,
                self.height,
                self.width * 4,
            )

    def draw_metadata(self, metadata, tracking_display):
        """Draw metadata and tracking points on the current frame."""
        if tracking_display:
            for point in tracking_display.history:
                self.draw_tracking_point(
                    point["center"], point["color"], point["opacity"]
                )

        for data in metadata:
            box = data["box"]
            self.draw_bounding_box(box)

            label = data.get("label", "")
            self.draw_text(label, box["x1"], box["y1"] - 10, Color(1, 0, 0, 1), 12)

            if tracking_display:
                track_id = data.get("track_id")
                if track_id is not None:
                    center = {
                        "x": (box["x1"] + box["x2"]) / 2,
                        "y": (box["y1"] + box["y2"]) / 2,
                    }
                    tracking_display.add_tracking_point(center, track_id)

    def finalize(self):
        """Finalize and clean up drawing."""
        if self.context:
            self.context.stroke()
        if self.surface:
            self.surface.finish()
        self.context = None
        self.surface = None

    def draw_bounding_box(self, box):
        self.context.set_source_rgb(1, 0, 0)
        self.context.set_line_width(2.0)
        self.context.rectangle(
            box["x1"], box["y1"], box["x2"] - box["x1"], box["y2"] - box["y1"]
        )
        self.context.stroke()

    def draw_text(self, label, x, y, color, font_size):
        self.context.set_source_rgba(color.b, color.g, color.r, color.a)
        self.context.set_font_size(font_size)
        self.context.move_to(x, y)
        self.context.show_text(label)
        self.context.stroke()

    def draw_tracking_point(self, center, color, opacity):
        size = 10
        half_size = size // 2
        self.context.set_source_rgba(color.b, color.g, color.r, opacity)
        self.context.rectangle(
            center["x"] - half_size, center["y"] - half_size, size, size
        )
        self.context.fill()

    def draw_line(self, start, end, color, width):
        self.context.set_source_rgba(color.b, color.g, color.r, color.a)
        self.context.set_line_width(width)
        self.context.move_to(start["x"], start["y"])
        self.context.line_to(end["x"], end["y"])
        self.context.stroke()
