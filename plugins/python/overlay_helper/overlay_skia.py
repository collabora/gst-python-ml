# overlay_skia.py
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import skia

from .overlay_utils_interface import (
    BOX_COLOR,
    BOX_LABEL_FONT_SIZE,
    BOX_LABEL_OFFSET,
    BOX_LINE_WIDTH,
    CLASSIFICATION_BACKGROUND_COLOR,
    CLASSIFICATION_BACKGROUND_MARGIN,
    CLASSIFICATION_FONT_SIZE,
    CLASSIFICATION_LABEL_WIDTH,
    CLASSIFICATION_LINE_HEIGHT,
    CLASSIFICATION_PADDING,
    CLASSIFICATION_TEXT_COLOR,
    TRACKING_POINT_SIZE,
    OverlayGraphics,
)

COLOR_TYPE_BY_VIDEO_FORMAT = {
    "RGBA": skia.ColorType.kRGBA_8888_ColorType,
    "BGRA": skia.ColorType.kBGRA_8888_ColorType,
}
BYTES_PER_PIXEL = 4


def to_skia_color(color, alpha):
    return skia.ColorSetARGB(
        round(alpha * 255),
        round(color.r * 255),
        round(color.g * 255),
        round(color.b * 255),
    )


class SkiaOverlayGraphics(OverlayGraphics):
    def __init__(self, width, height, video_format):
        if video_format not in COLOR_TYPE_BY_VIDEO_FORMAT:
            raise ValueError(
                f"skia overlay cannot draw on {video_format} frames, "
                f"supported formats: {', '.join(COLOR_TYPE_BY_VIDEO_FORMAT)}"
            )
        self.width = width
        self.height = height
        self.color_type = COLOR_TYPE_BY_VIDEO_FORMAT[video_format]
        # an empty family name picks the system default font
        self.typeface = skia.Typeface("")
        self.surface = None
        self.canvas = None

    def initialize(self, buffer_data):
        pixels = np.frombuffer(buffer_data, np.uint8).reshape(
            self.height, self.width, BYTES_PER_PIXEL
        )
        self.surface = skia.Surface(pixels, colorType=self.color_type)
        self.canvas = self.surface.getCanvas()

    def draw_metadata(self, metadata, tracking_display):
        if tracking_display:
            for point in tracking_display.history:
                self.draw_tracking_point(
                    point["center"], point["color"], point["opacity"]
                )

        classifications = [d for d in metadata if d.get("type") == "classification"]
        if classifications:
            self.draw_classification_labels(classifications)

        for data in metadata:
            if data.get("type") == "classification":
                continue
            box = data["box"]
            self.draw_bounding_box(box)

            label = data.get("label", "")
            self.draw_text(
                label,
                box["x1"],
                box["y1"] - BOX_LABEL_OFFSET,
                BOX_COLOR,
                BOX_LABEL_FONT_SIZE,
            )

            if tracking_display:
                track_id = data.get("track_id")
                if track_id is not None:
                    center = {
                        "x": (box["x1"] + box["x2"]) / 2,
                        "y": (box["y1"] + box["y2"]) / 2,
                    }
                    tracking_display.add_tracking_point(center, track_id)

    def draw_classification_labels(self, classifications):
        background = skia.Paint(
            Color=to_skia_color(
                CLASSIFICATION_BACKGROUND_COLOR, CLASSIFICATION_BACKGROUND_COLOR.a
            ),
            AntiAlias=True,
        )
        for index, item in enumerate(classifications):
            text = f"{item['label']}: {item['confidence']:.1%}"
            x = CLASSIFICATION_PADDING
            y = (
                CLASSIFICATION_PADDING
                + CLASSIFICATION_LINE_HEIGHT
                + index * CLASSIFICATION_LINE_HEIGHT
            )
            self.canvas.drawRect(
                skia.Rect.MakeXYWH(
                    x - CLASSIFICATION_BACKGROUND_MARGIN,
                    y - CLASSIFICATION_LINE_HEIGHT + CLASSIFICATION_BACKGROUND_MARGIN,
                    CLASSIFICATION_LABEL_WIDTH,
                    CLASSIFICATION_LINE_HEIGHT,
                ),
                background,
            )
            self.draw_text(
                text, x, y, CLASSIFICATION_TEXT_COLOR, CLASSIFICATION_FONT_SIZE
            )

    def finalize(self):
        self.surface.flushAndSubmit()
        self.canvas = None
        self.surface = None

    def draw_bounding_box(self, box):
        paint = skia.Paint(
            Color=to_skia_color(BOX_COLOR, BOX_COLOR.a),
            StrokeWidth=BOX_LINE_WIDTH,
            Style=skia.Paint.kStroke_Style,
            AntiAlias=True,
        )
        self.canvas.drawRect(
            skia.Rect.MakeLTRB(box["x1"], box["y1"], box["x2"], box["y2"]), paint
        )

    def draw_text(self, label, x, y, color, font_size):
        paint = skia.Paint(Color=to_skia_color(color, color.a), AntiAlias=True)
        self.canvas.drawString(label, x, y, skia.Font(self.typeface, font_size), paint)

    def draw_tracking_point(self, center, color, opacity):
        half_size = TRACKING_POINT_SIZE // 2
        paint = skia.Paint(Color=to_skia_color(color, opacity), AntiAlias=True)
        self.canvas.drawRect(
            skia.Rect.MakeXYWH(
                center["x"] - half_size,
                center["y"] - half_size,
                TRACKING_POINT_SIZE,
                TRACKING_POINT_SIZE,
            ),
            paint,
        )

    def draw_line(self, start, end, color, width):
        paint = skia.Paint(
            Color=to_skia_color(color, color.a),
            StrokeWidth=width,
            Style=skia.Paint.kStroke_Style,
            AntiAlias=True,
        )
        self.canvas.drawLine(start["x"], start["y"], end["x"], end["y"], paint)
