# overlay_utils_cairo.py
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

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

        classifications = [d for d in metadata if d.get("type") == "classification"]
        if classifications:
            self._draw_classification_labels(classifications)

        for data in metadata:
            if data.get("type") == "classification":
                continue
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

    def _draw_classification_labels(self, classifications):
        """Draw zero-shot classification results as corner text."""
        pad = 10
        line_h = 22
        font_size = 14
        label_w = 220

        for i, item in enumerate(classifications):
            text = f"{item['label']}: {item['confidence']:.1%}"
            x = pad
            y = pad + line_h + i * line_h
            # Dark background for readability
            self.context.set_source_rgba(0, 0, 0, 0.6)
            self.context.rectangle(x - 4, y - line_h + 4, label_w, line_h)
            self.context.fill()
            self.draw_text(text, x, y, Color(0, 1, 0, 1), font_size)

    def finalize(self):
        """Finalize and clean up drawing."""
        if self.context:
            self.context.stroke()
        if self.surface:
            self.surface.finish()
        self.context = None
        self.surface = None

    def draw_bounding_box(self, box):
        # the cairo surface is bgra, red is the third channel
        self.context.set_source_rgb(0, 0, 1)
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
