# overlay_utils_vulkan.py
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
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstVulkan", "1.0")
    from gi.repository import Gst, GstVulkan
except ImportError as e:
    raise ImportError(f"Failed to import Vulkan libraries: {e}")

import numpy as np
from .overlay_utils_interface import OverlayGraphics, Color


class VulkanOverlayGraphics(OverlayGraphics):
    def __init__(self, width, height, vk_device=None, vk_queue=None):
        self.width = width
        self.height = height
        self.vk_device = vk_device  # GstVulkanDevice
        self.vk_queue = vk_queue  # GstVulkanQueue
        self.vk_command_pool = None
        self.vk_command_buffer = None
        self.vk_image = None
        self.vk_image_view = None
        self.vk_render_pass = None
        self.vk_framebuffer = None
        self.vk_pipeline = None
        self.vk_pipeline_layout = None
        self.vk_vertex_buffer = None
        self.vertices = []  # List to store vertex data for batching
        self.colors = []  # List to store color data for batching
        self.setup_vulkan()

    def setup_vulkan(self):
        """Setup Vulkan resources for rendering."""
        if not self.vk_device:
            raise RuntimeError("Vulkan device not provided")
        if not self.vk_queue:
            raise RuntimeError("Vulkan queue not provided")

        # Create a command pool
        self.vk_command_pool = GstVulkan.CommandPool.new(
            self.vk_device, self.vk_queue.queue_family
        )
        if not self.vk_command_pool:
            raise RuntimeError("Failed to create Vulkan command pool")

        # Create a command buffer
        self.vk_command_buffer = self.vk_command_pool.create_buffer()
        if not self.vk_command_buffer:
            raise RuntimeError("Failed to create Vulkan command buffer")

        # Create a pipeline layout (simplified, no descriptor sets for now)
        self.vk_pipeline_layout = GstVulkan.PipelineLayout.new(self.vk_device)
        if not self.vk_pipeline_layout:
            raise RuntimeError("Failed to create Vulkan pipeline layout")

        # Create a vertex buffer
        # We'll allocate enough space for a reasonable number of vertices (e.g., 1000 quads = 4000 vertices)
        vertex_buffer_size = 4000 * 6 * 4  # 6 floats per vertex (x, y, r, g, b, a)
        self.vk_vertex_buffer = GstVulkan.Buffer.new(
            self.vk_device,
            vertex_buffer_size,
            GstVulkan.MemoryPropertyFlagBits.HOST_VISIBLE_BIT
            | GstVulkan.MemoryPropertyFlagBits.HOST_COHERENT_BIT,
        )
        if not self.vk_vertex_buffer:
            raise RuntimeError("Failed to create Vulkan vertex buffer")

    def cleanup(self):
        """Clean up Vulkan resources."""
        if self.vk_vertex_buffer:
            self.vk_vertex_buffer = None
        if self.vk_pipeline:
            self.vk_pipeline = None
        if self.vk_pipeline_layout:
            self.vk_pipeline_layout = None
        if self.vk_framebuffer:
            self.vk_framebuffer = None
        if self.vk_render_pass:
            self.vk_render_pass = None
        if self.vk_image_view:
            self.vk_image_view = None
        if self.vk_image:
            self.vk_image = None
        if self.vk_command_buffer:
            self.vk_command_buffer = None
        if self.vk_command_pool:
            self.vk_command_pool = None

    def create_render_pass(self):
        """Create a Vulkan render pass."""
        attachment = GstVulkan.AttachmentDescription()
        attachment.format = GstVulkan.Format.R8G8B8A8_UNORM
        attachment.load_op = (
            GstVulkan.AttachmentLoadOp.LOAD
        )  # Preserve existing content
        attachment.store_op = GstVulkan.AttachmentStoreOp.STORE
        attachment.initial_layout = GstVulkan.ImageLayout.TRANSFER_SRC_OPTIMAL
        attachment.final_layout = GstVulkan.ImageLayout.TRANSFER_SRC_OPTIMAL

        subpass = GstVulkan.SubpassDescription()
        subpass.add_color_attachment(0)

        render_pass_info = GstVulkan.RenderPassCreateInfo()
        render_pass_info.add_attachment(attachment)
        render_pass_info.add_subpass(subpass)

        return GstVulkan.RenderPass.new(self.vk_device, render_pass_info)

    def create_pipeline(self):
        """Create a Vulkan graphics pipeline (placeholder for shaders)."""
        # Note: This is a placeholder. In a real implementation, you would:
        # - Load SPIR-V vertex and fragment shaders
        # - Define vertex input state (position and color attributes)
        # - Set up viewport, rasterization, and blending states
        pipeline_info = GstVulkan.GraphicsPipelineCreateInfo()
        pipeline_info.render_pass = self.vk_render_pass
        pipeline_info.layout = self.vk_pipeline_layout

        # Placeholder for shader stages
        # You would need to create vertex and fragment shaders in SPIR-V format
        # For example:
        # vertex_shader = GstVulkan.Shader.new_from_spirv(self.vk_device, vertex_spirv_data)
        # fragment_shader = GstVulkan.Shader.new_from_spirv(self.vk_device, fragment_spirv_data)
        # pipeline_info.add_shader_stage(vertex_shader, GstVulkan.ShaderStage.VERTEX)
        # pipeline_info.add_shader_stage(fragment_shader, GstVulkan.ShaderStage.FRAGMENT)

        return GstVulkan.GraphicsPipeline.new(self.vk_device, pipeline_info)

    def initialize(self, buffer_data):
        """Initialize Vulkan context for rendering."""
        # Check if buffer_data is a GstBuffer with VulkanMemory
        if not isinstance(buffer_data, Gst.Buffer) or buffer_data.n_memory() == 0:
            raise RuntimeError("Buffer data must be a GstBuffer with memory")

        memory = buffer_data.peek_memory(0)
        if not GstVulkan.is_vulkan_memory(memory):
            raise RuntimeError("Buffer is not in VulkanMemory")

        # Get the Vulkan image from the buffer
        self.vk_image = GstVulkan.image_from_memory(memory)
        if not self.vk_image:
            raise RuntimeError("Failed to get Vulkan image from buffer")

        # Create an image view
        self.vk_image_view = GstVulkan.ImageView.new(
            self.vk_image,
            GstVulkan.ImageViewType.VIEW_2D,
            GstVulkan.Format.R8G8B8A8_UNORM,
        )
        if not self.vk_image_view:
            raise RuntimeError("Failed to create Vulkan image view")

        # Create a render pass
        self.vk_render_pass = self.create_render_pass()
        if not self.vk_render_pass:
            raise RuntimeError("Failed to create Vulkan render pass")

        # Create a framebuffer
        self.vk_framebuffer = GstVulkan.Framebuffer.new(
            self.vk_device,
            self.width,
            self.height,
            self.vk_render_pass,
            [self.vk_image_view],
        )
        if not self.vk_framebuffer:
            raise RuntimeError("Failed to create Vulkan framebuffer")

        # Create a pipeline
        self.vk_pipeline = self.create_pipeline()
        if not self.vk_pipeline:
            raise RuntimeError("Failed to create Vulkan pipeline")

        # Begin the command buffer
        self.vk_command_buffer.begin()

        # Begin the render pass
        clear_color = GstVulkan.ClearColorValue()
        clear_color.float32 = [0.0, 0.0, 0.0, 0.0]  # Transparent clear
        self.vk_command_buffer.begin_render_pass(
            self.vk_render_pass, self.vk_framebuffer, clear_color=clear_color
        )

        # Set up viewport and scissor
        viewport = GstVulkan.Viewport()
        viewport.x = 0
        viewport.y = 0
        viewport.width = self.width
        viewport.height = self.height
        viewport.min_depth = 0.0
        viewport.max_depth = 1.0
        self.vk_command_buffer.set_viewport(0, [viewport])

        scissor = GstVulkan.Rect2D()
        scissor.offset.x = 0
        scissor.offset.y = 0
        scissor.extent.width = self.width
        scissor.extent.height = self.height
        self.vk_command_buffer.set_scissor(0, [scissor])

        # Clear the vertex and color lists for the new frame
        self.vertices = []
        self.colors = []

    def draw_metadata(self, metadata, tracking_display):
        """Draw metadata and tracking points on the current frame."""
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

        # Flush all batched draw commands
        self.flush_draw()

    def flush_draw(self):
        """Flush all batched vertices and colors in a single draw call."""
        if not self.vertices:
            return  # Nothing to draw

        # Combine vertices and colors into a single array: [x, y, r, g, b, a]
        vertex_data = []
        for i in range(len(self.vertices) // 2):
            x = self.vertices[i * 2]
            y = self.vertices[i * 2 + 1]
            r = self.colors[i * 4]
            g = self.colors[i * 4 + 1]
            b = self.colors[i * 4 + 2]
            a = self.colors[i * 4 + 3]
            vertex_data.extend([x, y, r, g, b, a])

        # Convert to numpy array
        vertex_array = np.array(vertex_data, dtype=np.float32)

        # Upload vertex data to the buffer
        self.vk_vertex_buffer.write(vertex_array)

        # Bind the pipeline
        self.vk_command_buffer.bind_pipeline(self.vk_pipeline)

        # Bind the vertex buffer
        # Note: This requires setting up vertex input bindings in the pipeline
        # For simplicity, this is a placeholder
        # self.vk_command_buffer.bind_vertex_buffers(0, [self.vk_vertex_buffer], [0])

        # Draw the vertices
        # Note: This requires a proper pipeline with shaders
        # self.vk_command_buffer.draw(len(self.vertices) // 2, 1, 0, 0)

    def finalize(self):
        """Finalize Vulkan rendering and clean up."""
        # End the render pass
        self.vk_command_buffer.end_render_pass()

        # End the command buffer
        self.vk_command_buffer.end()

        # Submit the command buffer to the queue
        if self.vk_queue:
            self.vk_queue.submit(self.vk_command_buffer)

        # Clean up resources
        self.cleanup()

    def draw_bounding_box(self, box):
        """Add vertices for a bounding box to the batch (drawn as a quad outline)."""
        x1, y1 = box["x1"], box["y1"]
        x2, y2 = box["x2"], box["y2"]

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
        # Note: Vulkan doesn't have built-in text rendering.
        # You would need to pre-render text into a texture using a library like FreeType,
        # then draw textured quads here.
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
        dx = end["x"] - start["x"]
        dy = end["y"] - start["y"]
        length = (dx * dx + dy * dy) ** 0.5
        if length == 0:
            return

        dx /= length
        dy /= length

        perp_x = -dy
        perp_y = dx
        half_width = width / 2

        self.vertices.extend(
            [
                start["x"] + perp_x * half_width,
                start["y"] + perp_y * half_width,
                end["x"] + perp_x * half_width,
                end["y"] + perp_y * half_width,
                end["x"] - perp_x * half_width,
                end["y"] - perp_y * half_width,
                start["x"] - perp_x * half_width,
                start["y"] - perp_y * half_width,
            ]
        )

        self.colors.extend([color.r, color.g, color.b, color.a] * 4)
