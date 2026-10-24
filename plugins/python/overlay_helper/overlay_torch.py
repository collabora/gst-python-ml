# overlay_torch.py
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import math

from utils.format_converter import packed_format_layout

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

COLOUR_CHANNEL_COUNT = 3
FULL_INTENSITY = 255
TEXT_PIXELS_CACHE_SIZE = 256
# a box outline is four filled slabs
SLABS_PER_BOX = 4
# the 2d prefix sum of these corner marks is one inside the rectangle
CORNER_SIGNS = (1, -1, -1, 1)


class TorchOverlayGraphics(OverlayGraphics):
    def __init__(self, width, height, video_format):
        import torch

        self.width = width
        self.height = height
        channel_count, rgb_slice = packed_format_layout(video_format)
        rgb_channels = range(channel_count)[rgb_slice[-1]]
        first_channel = min(rgb_channels)
        self.colour_channels = slice(
            first_channel, first_channel + COLOUR_CHANNEL_COUNT
        )
        self.reverses_colour = rgb_channels.step < 0
        self.stream = torch.cuda.Stream()
        self.fonts = {}
        self.text_pixels_by_label = {}
        self.device_colours = {}
        self.pixels = None

    def drawing_stream(self):
        import torch

        return torch.cuda.stream(self.stream)

    def initialize(self, buffer_data):
        self.pixels = buffer_data[..., self.colour_channels]

    def finalize(self):
        # downstream waits on the producer's stream, never on this one
        self.stream.synchronize()
        self.pixels = None

    def draw_metadata(self, metadata, tracking_display):
        if tracking_display and tracking_display.history:
            self.draw_squares(
                [
                    self.square_row(point["center"], point["color"], point["opacity"])
                    for point in tracking_display.history
                ],
                TRACKING_POINT_SIZE,
            )

        classifications = [d for d in metadata if d.get("type") == "classification"]
        for index, item in enumerate(classifications):
            self.draw_classification_label(index, item)

        detections = [d for d in metadata if d.get("type") != "classification"]
        if not detections:
            return
        self.draw_bounding_boxes([data["box"] for data in detections])
        self.draw_texts(
            [
                (data.get("label", ""), data["box"]["x1"], data["box"]["y1"])
                for data in detections
            ],
            BOX_COLOR,
            BOX_LABEL_FONT_SIZE,
            BOX_LABEL_OFFSET,
        )
        if not tracking_display:
            return
        for data in detections:
            track_id = data.get("track_id")
            if track_id is None:
                continue
            box = data["box"]
            center = {
                "x": (box["x1"] + box["x2"]) / 2,
                "y": (box["y1"] + box["y2"]) / 2,
            }
            tracking_display.add_tracking_point(center, track_id)

    def draw_classification_label(self, index, item):
        x = CLASSIFICATION_PADDING
        y = CLASSIFICATION_PADDING + (index + 1) * CLASSIFICATION_LINE_HEIGHT
        left = x - CLASSIFICATION_BACKGROUND_MARGIN
        top = y - CLASSIFICATION_LINE_HEIGHT + CLASSIFICATION_BACKGROUND_MARGIN
        self.blend_rectangle(
            left,
            top,
            left + CLASSIFICATION_LABEL_WIDTH,
            top + CLASSIFICATION_LINE_HEIGHT,
            CLASSIFICATION_BACKGROUND_COLOR,
        )
        text = f"{item['label']}: {item['confidence']:.1%}"
        self.draw_text(text, x, y, CLASSIFICATION_TEXT_COLOR, CLASSIFICATION_FONT_SIZE)

    def draw_bounding_box(self, box):
        self.draw_bounding_boxes([box])

    def draw_bounding_boxes(self, boxes):
        import numpy as np
        import torch

        corners = np.array(
            [(box["x1"], box["y1"], box["x2"], box["y2"]) for box in boxes]
        )
        half_width = BOX_LINE_WIDTH // 2
        left, top = (corners[:, :2].round().astype(np.int64) - half_width).T
        right, bottom = (corners[:, 2:].round().astype(np.int64) + half_width).T
        slab_tops = np.concatenate([top, bottom - BOX_LINE_WIDTH, top, top])
        slab_bottoms = np.concatenate([top + BOX_LINE_WIDTH, bottom, bottom, bottom])
        slab_lefts = np.concatenate([left, left, left, right - BOX_LINE_WIDTH])
        slab_rights = np.concatenate([right, right, left + BOX_LINE_WIDTH, right])
        slab_tops = slab_tops.clip(0, self.height)
        slab_bottoms = np.maximum(slab_bottoms.clip(0, self.height), slab_tops)
        slab_lefts = slab_lefts.clip(0, self.width)
        slab_rights = np.maximum(slab_rights.clip(0, self.width), slab_lefts)
        rows = np.concatenate([slab_tops, slab_tops, slab_bottoms, slab_bottoms])
        columns = np.concatenate([slab_lefts, slab_rights, slab_lefts, slab_rights])
        signs = np.repeat(CORNER_SIGNS, len(boxes) * SLABS_PER_BOX)
        table = torch.from_numpy(np.stack([rows, columns, signs])).to(
            self.pixels.device
        )
        corner_marks = torch.zeros(
            self.height + 1, self.width + 1, dtype=torch.int32, device=table.device
        )
        corner_marks.index_put_(
            (table[0], table[1]), table[2].to(torch.int32), accumulate=True
        )
        coverage = corner_marks.cumsum(0, dtype=torch.int32).cumsum(
            1, dtype=torch.int32
        )
        covered = coverage[: self.height, : self.width, None] > 0
        colour = self.device_colour(BOX_COLOR).to(torch.uint8)
        self.pixels.copy_(torch.where(covered, colour, self.pixels))

    def draw_text(self, label, x, y, color, font_size):
        self.draw_texts([(label, x, y)], color, font_size, 0)

    def draw_texts(self, labels, color, font_size, baseline_offset):
        import numpy as np
        import torch

        pieces = []
        for text, x, y in labels:
            if not text:
                continue
            text_pixels, ascent = self.text_pixels(text, font_size)
            offset = np.array([[round(y) - baseline_offset - ascent], [round(x)], [0]])
            pieces.append(text_pixels + offset)
        if not pieces:
            return
        table = torch.from_numpy(np.concatenate(pieces, axis=1)).to(
            self.pixels.device, torch.float32
        )
        lines, columns, alphas = table
        colours = self.device_colour(color).expand(len(alphas), COLOUR_CHANNEL_COUNT)
        self.draw_pixels(lines.long(), columns.long(), colours, alphas)

    def draw_tracking_point(self, center, color, opacity):
        self.draw_squares(
            [self.square_row(center, color, opacity)], TRACKING_POINT_SIZE
        )

    def draw_line(self, start, end, color, width):
        self.draw_segments([(start, end, color)], width)

    def draw_polyline(self, points, color, width):
        pairs = list(zip(points, points[1:])) or [(points[0], points[0])]
        self.draw_segments([(start, end, color) for start, end in pairs], width)

    def draw_segments(self, segments, width):
        import numpy as np

        rows = []
        for start, end, color in segments:
            first = np.array((start["x"], start["y"]), float)
            last = np.array((end["x"], end["y"]), float)
            steps = max(1, math.ceil(np.abs(last - first).max()))
            fractions = np.arange(0, steps + 1)[:, None] / steps
            rows.extend(
                self.square_row({"x": x, "y": y}, color, color.a)
                for x, y in first + (last - first) * fractions
            )
        self.draw_squares(rows, width)

    def square_row(self, center, color, alpha):
        return (center["x"], center["y"], color.r, color.g, color.b, alpha)

    def draw_squares(self, rows, size):
        import numpy as np
        import torch

        if not rows:
            return
        device = self.pixels.device
        table = torch.from_numpy(np.array(rows, dtype=np.float32)).to(device)
        centre_columns, centre_lines, colours, alphas = table.split(
            [1, 1, COLOUR_CHANNEL_COUNT, 1], dim=1
        )
        channel_values = colours * FULL_INTENSITY
        if self.reverses_colour:
            channel_values = channel_values.flip(1)
        offsets = torch.arange(size, device=device) - size // 2
        columns = centre_columns.round().long()[:, :, None] + offsets[None, None, :]
        lines = centre_lines.round().long()[:, :, None] + offsets[None, :, None]
        columns, lines = torch.broadcast_tensors(columns, lines)
        pixels_per_square = size * size
        self.draw_pixels(
            lines.reshape(-1),
            columns.reshape(-1),
            channel_values.repeat_interleave(pixels_per_square, dim=0),
            alphas.repeat_interleave(pixels_per_square),
        )

    # only the last entry over a pixel is blended
    def draw_pixels(self, lines, columns, colours, alphas):
        import torch

        inside = (
            (columns >= 0)
            & (columns < self.width)
            & (lines >= 0)
            & (lines < self.height)
        )
        lines = lines.clamp(0, self.height - 1)
        columns = columns.clamp(0, self.width - 1)
        pixel_index = lines * self.width + columns
        # zero marks a pixel no entry inside the frame covers
        entry_number = torch.arange(1, len(lines) + 1, device=lines.device) * inside
        topmost = torch.zeros(
            self.width * self.height, dtype=torch.long, device=lines.device
        )
        topmost.scatter_reduce_(0, pixel_index, entry_number, "amax")
        winner = topmost[pixel_index]
        colours = torch.cat([colours.new_zeros(1, COLOUR_CHANNEL_COUNT), colours])
        alphas = torch.cat([alphas.new_zeros(1), alphas.reshape(-1)])
        background = self.pixels[lines, columns].float()
        blended = background.lerp_(colours[winner], alphas[winner, None])
        self.pixels[lines, columns] = blended.round_().to(torch.uint8)

    def blend_rectangle(self, left, top, right, bottom, color):
        rows = slice(max(top, 0), min(bottom, self.height))
        columns = slice(max(left, 0), min(right, self.width))
        if rows.start >= rows.stop or columns.start >= columns.stop:
            return
        region = self.pixels[rows, columns]
        blended = region.float().lerp_(self.device_colour(color), color.a)
        region.copy_(blended.round_())

    def channel_values(self, color):
        values = [
            color.r * FULL_INTENSITY,
            color.g * FULL_INTENSITY,
            color.b * FULL_INTENSITY,
        ]
        return values[::-1] if self.reverses_colour else values

    def device_colour(self, color):
        import torch

        key = (color.r, color.g, color.b)
        if key not in self.device_colours:
            self.device_colours[key] = torch.tensor(
                self.channel_values(color),
                dtype=torch.float32,
                device=self.pixels.device,
            )
        return self.device_colours[key]

    def font(self, font_size):
        from PIL import ImageFont

        if font_size not in self.fonts:
            self.fonts[font_size] = ImageFont.load_default(size=font_size)
        return self.fonts[font_size]

    # rows of line, column and alpha for each inked pixel, with the font ascent
    def text_pixels(self, text, font_size):
        key = (text, font_size)
        if key in self.text_pixels_by_label:
            return self.text_pixels_by_label[key]
        import numpy as np
        from PIL import Image, ImageDraw

        font = self.font(font_size)
        ascent, descent = font.getmetrics()
        image = Image.new("L", (math.ceil(font.getlength(text)), ascent + descent))
        ImageDraw.Draw(image).text((0, 0), text, fill=FULL_INTENSITY, font=font)
        alpha = np.asarray(image)
        lines, columns = np.nonzero(alpha)
        inked = np.stack([lines, columns, alpha[lines, columns] / FULL_INTENSITY])
        if len(self.text_pixels_by_label) >= TEXT_PIXELS_CACHE_SIZE:
            self.text_pixels_by_label.clear()
        self.text_pixels_by_label[key] = (inked.astype(np.float32), ascent)
        return self.text_pixels_by_label[key]
