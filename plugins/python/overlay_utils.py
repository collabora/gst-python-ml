# OverlayUtils
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

import os
import json
import cairo
from abc import ABC, abstractmethod
from enum import Enum

# Add OpenGL imports for the new OpenGL-based overlay graphics
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
except ImportError as e:
    raise ImportError(f"Failed to import OpenGL libraries: {e}")


class Color:
    def __init__(self, r, g, b, a=1.0):
        self.r = r
        self.g = g
        self.b = b
        self.a = a


class TrackingDisplay:
    def __init__(self, max_history_length=5000):
        self.history = []
        self.max_history_length = max_history_length
        self.id_color_map = {}
        self.color_palette = [
            Color(1.0, 0.0, 0.0, 1.0),  # Red
            Color(0.0, 1.0, 0.0, 1.0),  # Green
            Color(0.0, 0.0, 1.0, 1.0),  # Blue
            Color(1.0, 1.0, 0.0, 1.0),  # Yellow
            Color(1.0, 0.0, 1.0, 1.0),  # Magenta
            Color(0.0, 1.0, 1.0, 1.0),  # Cyan
            Color(1.0, 0.5, 0.0, 1.0),  # Orange
            Color(0.5, 0.0, 1.0, 1.0),  # Purple
            Color(0.5, 1.0, 0.0, 1.0),  # Lime
            Color(0.0, 0.5, 1.0, 1.0),  # Light Blue
            Color(1.0, 0.3, 0.3, 1.0),  # Light Red
            Color(0.3, 1.0, 0.3, 1.0),  # Light Green
            Color(0.3, 0.3, 1.0, 1.0),  # Light Blue
            Color(1.0, 1.0, 0.3, 1.0),  # Light Yellow
            Color(1.0, 0.3, 1.0, 1.0),  # Pink
            Color(0.3, 1.0, 1.0, 1.0),  # Aqua
            Color(0.5, 0.2, 0.0, 1.0),  # Brown
            Color(0.2, 0.5, 0.0, 1.0),  # Olive
            Color(0.5, 0.5, 0.5, 1.0),  # Grey
            Color(1.0, 0.6, 0.4, 1.0),  # Peach
        ]
        self.processed_ids = set()  # Track IDs that have already been counted
        self.total_top_to_bottom = 0  # Total objects crossing from top to bottom
        self.total_bottom_to_top = 0  # Total objects crossing from bottom to top
        self.y_line = None  # The y-coordinate of the horizontal line

    def get_color_for_id(self, track_id):
        if track_id not in self.id_color_map:
            color_index = len(self.id_color_map) % len(self.color_palette)
            self.id_color_map[track_id] = self.color_palette[color_index]
        return self.id_color_map[track_id]

    def add_tracking_point(self, center, track_id):
        color = self.get_color_for_id(track_id)
        self.history.append(
            {"center": center, "color": color, "track_id": track_id, "opacity": 1.0}
        )

        if len(self.history) > self.max_history_length:
            self.history = self.history[-self.max_history_length :]

    def fade_history(self):
        for point in self.history:
            point["opacity"] *= 0.9
        self.history = [point for point in self.history if point["opacity"] > 0.1]

    def set_y_line(self, y_line):
        """Set the y-coordinate of the horizontal line for calculations."""
        self.y_line = y_line

    def count_objects(self):
        """Count cars crossing the horizontal line set by `set_y_line`.

        Returns:
            tuple: (count_top_to_bottom, count_bottom_to_top)
        """
        if self.y_line is None:
            raise ValueError("The y_line must be set before counting crossings.")

        count_top_to_bottom = 0
        count_bottom_to_top = 0

        track_last_positions = {}

        for point in self.history:
            track_id = point["track_id"]
            if track_id in self.processed_ids:
                continue  # Skip tracks that have already been processed

            current_y = point["center"]["y"]

            if track_id in track_last_positions:
                last_y = track_last_positions[track_id]

                if last_y < self.y_line <= current_y:  # Crossed from top to bottom
                    count_top_to_bottom += 1
                    self.total_top_to_bottom += 1
                    self.processed_ids.add(track_id)
                elif last_y > self.y_line >= current_y:  # Crossed from bottom to top
                    count_bottom_to_top += 1
                    self.total_bottom_to_top += 1
                    self.processed_ids.add(track_id)

            track_last_positions[track_id] = current_y

        return self.total_top_to_bottom, self.total_bottom_to_top


class OverlayGraphics(ABC):
    @abstractmethod
    def initialize(self, buffer_data):
        """Initialize the graphics context and prepare for rendering."""
        pass

    @abstractmethod
    def draw_metadata(self, metadata, tracking_display):
        """Draw metadata and tracking points on the current frame."""
        pass

    @abstractmethod
    def finalize(self):
        """Finalize the rendering and clean up the context."""
        pass

    @abstractmethod
    def draw_bounding_box(self, box):
        """Draw a bounding box on the current frame."""
        pass

    @abstractmethod
    def draw_text(self, label, x, y, colour, font_size):
        """Draw a label at the specified position on the current frame."""
        pass

    @abstractmethod
    def draw_tracking_point(self, center, color, opacity):
        """Draw a tracking point with specified color and opacity."""
        pass

    @abstractmethod
    def draw_line(self, start, end, color, width):
        """Draw a line from start to end with specified color and width."""
        pass


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


class OpenGLOverlayGraphics(OverlayGraphics):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.fbo = None
        self.texture = None
        self.setup_opengl()

    def setup_opengl(self):
        """Setup OpenGL resources like FBO and texture for offscreen rendering."""
        # Generate and bind a Framebuffer Object (FBO)
        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)

        # Generate and bind a texture for the FBO
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA,
            self.width,
            self.height,
            0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            None,
        )
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        # Attach the texture to the FBO
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.texture, 0
        )

        # Check if the FBO is complete
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Framebuffer setup failed")

        # Unbind the FBO
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def initialize(self, buffer_data):
        """Initialize OpenGL context for rendering."""
        # Bind the FBO for offscreen rendering
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.width, self.height)

        # Clear the buffer
        glClearColor(0.0, 0.0, 0.0, 0.0)  # Clear with transparent background
        glClear(GL_COLOR_BUFFER_BIT)

        # Setup orthographic projection to match the video frame coordinates
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0, self.width, self.height, 0)  # Flip y-axis to match video
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

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
        """Finalize OpenGL rendering and clean up."""
        # Unbind the FBO and flush the OpenGL commands
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glFlush()

    def draw_bounding_box(self, box):
        """Draw a bounding box using OpenGL."""
        glColor4f(1.0, 0.0, 0.0, 1.0)  # Red color
        glLineWidth(2.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(box["x1"], box["y1"])
        glVertex2f(box["x2"], box["y1"])
        glVertex2f(box["x2"], box["y2"])
        glVertex2f(box["x1"], box["y2"])
        glEnd()

    def draw_text(self, label, x, y, color, font_size):
        """Draw text at the specified position (placeholder for texture atlas)."""
        # Note: OpenGL doesn't have built-in text rendering.
        # In a production environment, you should pre-render text into a texture atlas
        # using a library like FreeType, then draw textured quads here.
        # For now, this is a placeholder.
        pass

    def draw_tracking_point(self, center, color, opacity):
        """Draw a tracking point with specified color and opacity."""
        size = 10
        half_size = size / 2
        glColor4f(color.r, color.g, color.b, opacity)
        glBegin(GL_QUADS)
        glVertex2f(center["x"] - half_size, center["y"] - half_size)
        glVertex2f(center["x"] + half_size, center["y"] - half_size)
        glVertex2f(center["x"] + half_size, center["y"] + half_size)
        glVertex2f(center["x"] - half_size, center["y"] + half_size)
        glEnd()

    def draw_line(self, start, end, color, width):
        """Draw a line from start to end with specified color and width."""
        glColor4f(color.r, color.g, color.b, color.a)
        glLineWidth(width)
        glBegin(GL_LINES)
        glVertex2f(start["x"], start["y"])
        glVertex2f(end["x"], end["y"])
        glEnd()


class GraphicsType(Enum):
    CAIRO = ("cairo",)
    SKIA = "skia"
    OPENGL = "opengl"  # Add OpenGL as a graphics type


class OverlayGraphicsFactory:
    @staticmethod
    def create(graphics_type, width, height):
        """Factory method to create an OverlayGraphics object based on type."""
        if graphics_type == GraphicsType.CAIRO:
            return CairoOverlayGraphics(width, height)
        elif graphics_type == GraphicsType.OPENGL:
            return OpenGLOverlayGraphics(width, height)
        else:
            raise ValueError(f"Unknown graphics type: {graphics_type}")


def load_metadata(meta_path, logger):
    """Load JSON metadata from a file and return a dictionary indexed by frame index.

    Args:
        meta_path (str): Path to the JSON metadata file.

    Returns:
        dict: Metadata indexed by frame index.
    """
    if not meta_path:
        logger.error("Frame metadata file path not set.")
        return {}

    if not os.path.exists(meta_path):
        logger.error(f"JSON file not found: {meta_path}")
        return {}

    try:
        with open(meta_path, "r") as f:
            all_data = json.load(f)
            frame_data = all_data.get("frames", [])
            # Store metadata indexed by frame_index
            metadata = {
                frame.get("frame_index"): frame.get("objects", [])
                for frame in frame_data
            }
            logger.info(f"Loaded metadata for {len(metadata)} frames.")
            return metadata
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON file: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error while loading metadata: {e}")
        return {}
