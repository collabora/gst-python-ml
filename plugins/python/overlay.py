# Overlay
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

from analytics_utils import ANALYTICS_UTILS_AVAILABLE

if ANALYTICS_UTILS_AVAILABLE:
    from analytics_utils import AnalyticsUtils

CAN_REGISTER_ELEMENT = True
try:
    from overlay_utils import (
        load_metadata,
        TrackingDisplay,
        GraphicsType,
        OverlayGraphicsFactory,
    )
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    gi.require_version("GstVideo", "1.0")
    gi.require_version("GstGL", "1.0")  # Add OpenGL support
    from gi.repository import (
        Gst,
        GstBase,
        GstVideo,
        GstGL,
        GObject,
    )  # noqa: E402
    from log.logger_factory import LoggerFactory
except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'pyml_overlay' element will not be available. Error: {e}"
    )

# Support both CPU and GPU buffers
VIDEO_FORMATS = "video/x-raw, format=(string){ RGBA, ARGB, BGRA, ABGR }; video/x-raw(memory:GLMemory), format=(string){ RGBA, ARGB, BGRA, ABGR }"
OVERLAY_CAPS = Gst.Caps.from_string(VIDEO_FORMATS)


class Overlay(GstBase.BaseTransform):
    __gstmetadata__ = (
        "Overlay",
        "Filter/Effect/Video",
        "Overlays object detection / tracking data on video",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    src_template = Gst.PadTemplate.new(
        "src",
        Gst.PadDirection.SRC,
        Gst.PadPresence.ALWAYS,
        OVERLAY_CAPS.copy(),
    )

    sink_template = Gst.PadTemplate.new(
        "sink",
        Gst.PadDirection.SINK,
        Gst.PadPresence.ALWAYS,
        OVERLAY_CAPS.copy(),
    )
    __gsttemplates__ = (src_template, sink_template)

    meta_path = GObject.Property(
        type=str,
        default=None,
        nick="Metadata File Path",
        blurb="Path to the JSON file containing frame metadata with bounding boxes and tracking data",
        flags=GObject.ParamFlags.READWRITE,
    )
    tracking = GObject.Property(
        type=bool,
        default=True,
        nick="Enable Tracking Display",
        blurb="Enable or disable tracking display",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        self.logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)
        self.extracted_metadata = {}
        self.from_file = False
        self.frame_counter = 0
        self.tracking_display = TrackingDisplay()
        self.do_set_dims(0, 0)
        self.overlay_graphics = None
        self.graphics_type = None
        self.gl_context = None
        self.use_opengl = False
        self.gl_display = None
        self.context_set = False

    def do_get_property(self, prop: GObject.ParamSpec):
        if prop.name == "meta-path":
            return self.meta_path
        elif prop.name == "tracking":
            return self.tracking
        else:
            raise AttributeError(f"Unknown property {prop.name}")

    def do_set_property(self, prop: GObject.ParamSpec, value):
        if prop.name == "meta-path":
            self.meta_path = value
        elif prop.name == "tracking":
            self.tracking = value
            self.logger.info(f"Tracking set to: {self.tracking}")
        else:
            raise AttributeError(f"Unknown property {prop.name}")

    def on_message(self, bus, message):
        if message.type == Gst.MessageType.EOS:
            self.logger.info("Reset frame counter.")
            self.frame_counter = 0
        elif message.type == Gst.MessageType.NEED_CONTEXT:
            self.logger.info("Received NEED_CONTEXT message")
            context = message.parse_context()
            if context and context.get_context_type() == "gst.gl.app_context":
                self.gl_context = context.get_gl_context()
                if self.gl_context:
                    self.logger.info(
                        "Successfully acquired OpenGL context from NEED_CONTEXT"
                    )
                    self.use_opengl = True
                    self.context_set = True
                else:
                    self.logger.warning(
                        "Failed to get OpenGL context from NEED_CONTEXT"
                    )
            elif context and context.get_context_type() == "gst.gl.display":
                self.gl_display = context.get_gl_display()
                if self.gl_display:
                    self.logger.info(
                        "Successfully acquired GL display from NEED_CONTEXT"
                    )

    def do_start(self):
        if self.bus:
            self.bus.add_signal_watch()
            self.bus.connect("message", self.on_message)
            self.logger.info("Added signal watch to pipeline's bus.")
        else:
            self.logger.error("Could not get the bus from the pipeline.")
        return True

    def do_set_caps(self, incaps, outcaps):
        video_info = GstVideo.VideoInfo.new_from_caps(incaps)
        self.do_set_dims(video_info.width, video_info.height)
        self.logger.info(f"Video caps set: width={self.width}, height={self.height}")

        # Check if the input caps are using GLMemory
        self.use_opengl = "memory:GLMemory" in incaps.to_string()
        self.logger.info(f"Using OpenGL: {self.use_opengl}")

        # We’ll defer graphics backend initialization to do_transform_ip
        # after the OpenGL context is retrieved via NEED_CONTEXT
        return True

    def do_set_dims(self, width, height):
        self.width = width
        self.height = height

    def do_transform_ip(self, buf):
        # First try to extract metadata from frame meta
        if ANALYTICS_UTILS_AVAILABLE and not self.from_file:
            analytics_utils = AnalyticsUtils()
            extracted = analytics_utils.extract_analytics_metadata(buf)
            self.logger.debug(f"Extracted buffer metadata: {extracted}")
            if extracted:
                self.extracted_metadata = extracted
            else:
                self.logger.warning(
                    "No metadata extracted from buffer, checking file fallback"
                )

        # Fall back to file if buffer metadata is empty and meta_path is set
        if not self.extracted_metadata and self.meta_path:
            self.logger.info(f"Attempting to load metadata from file: {self.meta_path}")
            self.extracted_metadata = load_metadata(self.meta_path, self.logger)
            self.from_file = True

        # If no metadata from either source, pass through without overlay
        if not self.extracted_metadata:
            self.logger.info(
                "No metadata available from buffer or file, passing through buffer"
            )
            self.frame_counter += 1
            return Gst.FlowReturn.OK

        frame_metadata = None
        if self.from_file:
            frame_metadata = self.extracted_metadata.get(self.frame_counter, [])
            self.logger.info(
                f"Using file metadata for frame {self.frame_counter}: {frame_metadata}"
            )
        else:
            frame_metadata = self.extracted_metadata
            self.logger.debug(f"Using buffer metadata: {frame_metadata}")

        # Initialize the graphics backend if not already done
        if self.overlay_graphics is None:
            # If we expected OpenGL but the context isn't set, fall back to Cairo
            if self.use_opengl and not self.context_set:
                self.logger.warning("OpenGL context not set, falling back to Cairo")
                self.use_opengl = False

            # Check if the buffer's memory is GLMemory
            is_gl_buffer = False
            if buf.n_memory() > 0:  # Ensure the buffer has at least one memory object
                memory = buf.peek_memory(0)  # Get the first memory object
                is_gl_buffer = GstGL.is_gl_memory(memory)
            else:
                self.logger.warning(
                    "Buffer has no memory objects, falling back to Cairo"
                )
                is_gl_buffer = False

            # If the buffer is not GLMemory, force Cairo rendering
            if not is_gl_buffer:
                self.logger.info("Buffer is not GLMemory, using Cairo rendering")
                self.use_opengl = False

            self.graphics_type = (
                GraphicsType.OPENGL if self.use_opengl else GraphicsType.CAIRO
            )
            self.overlay_graphics = OverlayGraphicsFactory.create(
                self.graphics_type, self.width, self.height
            )
            self.logger.info(f"Initialized graphics backend: {self.graphics_type}")

        # Handle rendering based on the graphics type
        if self.graphics_type == GraphicsType.OPENGL:
            # Double-check that the buffer is GLMemory
            is_gl_buffer = False
            if buf.n_memory() > 0:
                memory = buf.peek_memory(0)
                is_gl_buffer = GstGL.is_gl_memory(memory)

            if not is_gl_buffer:
                self.logger.error(
                    "Buffer is not in GLMemory, cannot proceed with OpenGL overlay"
                )
                return Gst.FlowReturn.ERROR

            if not self.gl_context:
                self.logger.error(
                    "No OpenGL context available, cannot proceed with OpenGL overlay"
                )
                return Gst.FlowReturn.ERROR

            try:
                # Make the OpenGL context current
                self.gl_context.make_current()
                self.overlay_graphics.initialize(buf)
                self.do_post_process(frame_metadata)
                self.overlay_graphics.finalize()
            except Exception as e:
                self.logger.error(f"Error during OpenGL rendering: {e}")
                return Gst.FlowReturn.ERROR
            finally:
                self.gl_context.make_current(False)
        else:  # Cairo rendering
            video_meta = GstVideo.buffer_get_video_meta(buf)
            if not video_meta:
                self.logger.error(
                    "No video meta available, cannot proceed with overlay"
                )
                return Gst.FlowReturn.ERROR

            success, map_info = buf.map(Gst.MapFlags.WRITE)
            if not success:
                self.logger.error("Failed to map buffer for writing")
                return Gst.FlowReturn.ERROR

            try:
                self.overlay_graphics.initialize(map_info.data)
                self.do_post_process(frame_metadata)
                self.overlay_graphics.finalize()
            finally:
                buf.unmap(map_info)

        self.frame_counter += 1
        return Gst.FlowReturn.OK

    def do_post_process(self, frame_metadata):
        self.overlay_graphics.draw_metadata(
            frame_metadata, self.tracking_display if self.tracking else None
        )
        if self.tracking:
            self.tracking_display.fade_history()


if CAN_REGISTER_ELEMENT:
    GObject.type_register(Overlay)
    __gstelementfactory__ = ("pyml_overlay", Gst.Rank.NONE, Overlay)
else:
    GlobalLogger().warning(
        "The 'pyml_overlay' element will not be registered because a module is missing."
    )
