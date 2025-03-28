# overlay_utils_opengl.py
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

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    import numpy as np
except ImportError as e:
    raise ImportError(f"Failed to import OpenGL libraries: {e}")

from .overlay_utils_interface import OverlayGraphics, Color


class OpenGLOverlayGraphics(OverlayGraphics):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.fbo = None
        self.texture = None
        self.vbo = None
        self.vertices = []  # List to store vertex data for batching
        self.colors = []  # List to store color data for batching
        self.setup_opengl()

    def setup_opengl(self):
        """Setup OpenGL resources like FBO, texture, and VBO for offscreen rendering."""
        # Generate and bind a Framebuffer Object (FBO)
        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)

        # Print OpenGL version
        try:
            version = glGetString(GL_VERSION)
            if version:
                print(f"OpenGL Version: {version.decode('utf-8')}")
            else:
                print("Failed to retrieve OpenGL version: glGetString returned None")
        except Exception as e:
            print(f"Error retrieving OpenGL version: {e}")

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

        # Generate a Vertex Buffer Object (VBO) for batching
        self.vbo = glGenBuffers(1)

        # Unbind the FBO
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def cleanup(self):
        """Clean up OpenGL resources."""
        if self.fbo:
            glDeleteFramebuffers(1, [self.fbo])
        if self.texture:
            glDeleteTextures([self.texture])
        if self.vbo:
            glDeleteBuffers(1, [self.vbo])

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

        # Clear the vertex and color lists for the new frame
        self.vertices = []
        self.colors = []

    def draw_metadata(self, metadata, tracking_display):
        """Draw metadata and tracking points on the current frame."""
        # Clear the vertex and color lists for this frame
        self.vertices = []
        self.colors = []

        # Draw tracking points if tracking display is enabled
        if tracking_display:
            for point in tracking_display.history:
                self.draw_tracking_point(
                    point["center"], point["color"], point["opacity"]
                )

        # Draw bounding boxes and labels for each metadata object
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

        # Flush all batched draw commands in a single draw call
        self.flush_draw()

    def flush_draw(self):
        """Flush all batched vertices and colors in a single draw call."""
        if not self.vertices:
            return  # Nothing to draw

        # Convert vertices and colors to numpy arrays
        vertex_array = np.array(self.vertices, dtype=np.float32)
        color_array = np.array(self.colors, dtype=np.float32)

        # Bind the VBO
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

        # Upload vertex data
        glBufferData(GL_ARRAY_BUFFER, vertex_array.nbytes, vertex_array, GL_STATIC_DRAW)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(2, GL_FLOAT, 0, None)

        # Upload color data (using a separate buffer offset)
        glBufferData(GL_ARRAY_BUFFER, color_array.nbytes, color_array, GL_STATIC_DRAW)
        glEnableClientState(GL_COLOR_ARRAY)
        glColorPointer(4, GL_FLOAT, 0, None)

        # Draw all vertices in a single draw call
        glDrawArrays(GL_QUADS, 0, len(self.vertices) // 2)

        # Disable vertex and color arrays
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

        # Unbind the VBO
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def finalize(self):
        """Finalize OpenGL rendering and clean up."""
        # Unbind the FBO and flush the OpenGL commands
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glFlush()

    def draw_bounding_box(self, box):
        """Add vertices for a bounding box to the batch (drawn as a quad outline)."""
        # Define the four corners of the bounding box
        x1, y1 = box["x1"], box["y1"]
        x2, y2 = box["x2"], box["y2"]

        # Draw the bounding box as a quad outline by defining four line segments
        # Top line: (x1, y1) to (x2, y1)
        self.vertices.extend([x1, y1, x2, y1, x2, y1 + 1, x1, y1 + 1])
        self.colors.extend([1.0, 0.0, 0.0, 1.0] * 4)  # Red color

        # Right line: (x2, y1) to (x2, y2)
        self.vertices.extend([x2, y1, x2 + 1, y1, x2 + 1, y2, x2, y2])
        self.colors.extend([1.0, 0.0, 0.0, 1.0] * 4)

        # Bottom line: (x2, y2) to (x1, y2)
        self.vertices.extend([x2, y2, x1, y2, x1, y2 - 1, x2, y2 - 1])
        self.colors.extend([1.0, 0.0, 0.0, 1.0] * 4)

        # Left line: (x1, y2) to (x1, y1)
        self.vertices.extend([x1, y2, x1 - 1, y2, x1 - 1, y1, x1, y1])
        self.colors.extend([1.0, 0.0, 0.0, 1.0] * 4)

    def draw_text(self, label, x, y, color, font_size):
        """Draw text at the specified position (placeholder for texture atlas)."""
        # Note: OpenGL doesn't have built-in text rendering.
        # In a production environment, you should pre-render text into a texture atlas
        # using a library like FreeType, then draw textured quads here.
        # For now, this is a placeholder.
        pass

    def draw_tracking_point(self, center, color, opacity):
        """Add vertices for a tracking point to the batch (drawn as a quad)."""
        size = 10
        half_size = size / 2
        x, y = center["x"], center["y"]

        # Define the four corners of the quad
        self.vertices.extend(
            [
                x - half_size,
                y - half_size,  # Bottom-left
                x + half_size,
                y - half_size,  # Bottom-right
                x + half_size,
                y + half_size,  # Top-right
                x - half_size,
                y + half_size,  # Top-left
            ]
        )

        # Add the color for all four vertices
        self.colors.extend([color.r, color.g, color.b, opacity] * 4)

    def draw_line(self, start, end, color, width):
        """Add vertices for a line to the batch (drawn as a thin quad)."""
        # Calculate the direction vector of the line
        dx = end["x"] - start["x"]
        dy = end["y"] - start["y"]
        length = (dx * dx + dy * dy) ** 0.5
        if length == 0:
            return

        # Normalize the direction vector
        dx /= length
        dy /= length

        # Calculate the perpendicular vector for the line width
        perp_x = -dy
        perp_y = dx
        half_width = width / 2

        # Define the four corners of the quad representing the line
        self.vertices.extend(
            [
                start["x"] + perp_x * half_width,
                start["y"] + perp_y * half_width,  # Bottom-left
                end["x"] + perp_x * half_width,
                end["y"] + perp_y * half_width,  # Bottom-right
                end["x"] - perp_x * half_width,
                end["y"] - perp_y * half_width,  # Top-right
                start["x"] - perp_x * half_width,
                start["y"] - perp_y * half_width,  # Top-left
            ]
        )

        # Add the color for all four vertices
        self.colors.extend([color.r, color.g, color.b, color.a] * 4)
