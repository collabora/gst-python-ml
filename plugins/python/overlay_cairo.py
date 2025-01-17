# OverlayCairo
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
    from utils import load_metadata
    import gi
    import cairo

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    gi.require_version("GstVideo", "1.0")
    from gi.repository import (
        Gst,
        GstBase,
        GstVideo,
        GObject,
    )  # noqa: E402
except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    Gst.warning(f"The 'pyml_overlay' element will not be available. Error: {e}")

# Define video formats manually
VIDEO_FORMATS = "video/x-raw, format=(string){ RGBA, ARGB, BGRA, ABGR }"

# Create OBJECT_DETECTION_OVERLAY_CAPS
OBJECT_DETECTION_OVERLAY_CAPS = Gst.Caps.from_string(VIDEO_FORMATS)


class OverlayCairo(GstBase.BaseTransform):
    __gstmetadata__ = (
        "OverlayCairo",
        "Filter/Effect/Video",
        "Overlays object detection / tracking data on video",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    src_template = Gst.PadTemplate.new(
        "src",
        Gst.PadDirection.SRC,
        Gst.PadPresence.ALWAYS,
        OBJECT_DETECTION_OVERLAY_CAPS.copy(),
    )

    sink_template = Gst.PadTemplate.new(
        "sink",
        Gst.PadDirection.SINK,
        Gst.PadPresence.ALWAYS,
        OBJECT_DETECTION_OVERLAY_CAPS.copy(),
    )
    __gsttemplates__ = (src_template, sink_template)

    # Add the meta_path property
    meta_path = GObject.Property(
        type=str,
        default=None,
        nick="Metadata Path",
        blurb="Path to the JSON file containing frame metadata with bounding boxes and tracking data",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super(OverlayCairo, self).__init__()
        self.meta_path = None  # Initialize the frame_meta property
        self.preloaded_metadata = {}  # Dictionary to store frame-indexed metadata
        self.frame_counter = 0
        self.width = 640
        self.height = 480
        self.history = []
        self.max_history_length = 5000

        # Dictionary to store ID-to-color mapping
        self.id_color_map = {}

    def do_get_property(self, prop: GObject.ParamSpec):
        if prop.name == "meta-path":
            return self.meta_path
        else:
            raise AttributeError(f"Unknown property {prop.name}")

    def do_set_property(self, prop: GObject.ParamSpec, value):
        if prop.name == "meta-path":
            self.meta_path = value
            self.load_and_store_metadata()  # Load JSON data when property is set
        else:
            raise AttributeError(f"Unknown property {prop.name}")

    def on_message(self, bus, message):
        """Handle messages from the pipeline's bus."""
        if message.type == Gst.MessageType.SEGMENT_DONE:
            Gst.info("reset frame counter.")
            self.frame_counter = 0

    def do_start(self):
        if self.bus:
            self.bus.add_signal_watch()
            self.bus.connect("message", self.on_message)
            Gst.info("Added signal watch to pipeline's bus.")
        else:
            Gst.error("Could not get the bus from the pipeline.")
        return True

    def do_set_caps(self, incaps, outcaps):
        video_info = GstVideo.VideoInfo.new_from_caps(incaps)
        self.width = video_info.width
        self.height = video_info.height
        Gst.info(f"Video caps set: width={self.width}, height={self.height}")
        return True

    def get_metadata_for_frame(self, frame_index):
        """Retrieve preloaded metadata for the given frame index."""
        return self.preloaded_metadata.get(frame_index, [])

    def do_transform_ip(self, buf):
        if not self.preloaded_metadata:
            self.preloaded_metadata = load_metadata(self.meta_path)
        metadata = self.get_metadata_for_frame(self.frame_counter)

        # Skip processing if no metadata exists for the current frame
        if not metadata:
            Gst.warning(f"No metadata found for frame {self.frame_counter}.")
            self.frame_counter += 1
            return Gst.FlowReturn.OK

        video_meta = GstVideo.buffer_get_video_meta(buf)
        if not video_meta:
            Gst.error("No video meta available, cannot proceed with overlay")
            return Gst.FlowReturn.ERROR

        success, map_info = buf.map(Gst.MapFlags.WRITE)
        if not success:
            return Gst.FlowReturn.ERROR

        cairo_surface = None
        cr = None

        try:
            # Create a Cairo surface for text rendering directly on buffer data
            cairo_surface = cairo.ImageSurface.create_for_data(
                map_info.data,
                cairo.FORMAT_ARGB32,
                self.width,
                self.height,
                self.width * 4,
            )
            cr = cairo.Context(cairo_surface)

            # Draw bounding boxes and labels on main surface
            for data in metadata:
                # Draw bounding box
                self.draw_bounding_box_with_cairo(cr, data["box"])

                # Draw label near the bounding box using Cairo
                self.draw_label_with_cairo(
                    cr, data["label"], data["box"]["x1"], data["box"]["y1"]
                )

            # Ensure Cairo operations are complete before unmapping
            cr.stroke()
            cairo_surface.finish()

        finally:
            # Cleanup resources
            if cairo_surface:
                del cairo_surface
            if cr:
                del cr
            buf.unmap(map_info)  # Unmap buffer after ensuring no references remain
            self.frame_counter += 1

        return Gst.FlowReturn.OK

    def draw_bounding_box_with_cairo(self, cr, box):
        """Draw a bounding box using Cairo."""
        cr.set_line_width(2.0)
        cr.set_source_rgb(1, 0, 0)  # Red color for bounding box
        cr.rectangle(box["x1"], box["y1"], box["x2"] - box["x1"], box["y2"] - box["y1"])
        cr.stroke()

    def draw_label_with_cairo(self, cr, label, x, y):
        """Draw a label using Cairo at the specified position."""
        cr.set_font_size(12)
        cr.set_source_rgb(1, 1, 1)  # White color for label
        cr.move_to(x, y - 10)  # Position the text above the bounding box
        cr.show_text(label)
        cr.stroke()

    def draw_label_with_cairo(self, cr, label, x, y):
        """Draws a label with Cairo at the specified position."""
        cr.set_font_size(12)
        cr.set_source_rgba(1, 1, 1, 1)  # White color
        cr.move_to(x, y - 10)  # Position the text above the bounding box
        cr.show_text(label)
        cr.stroke()


if CAN_REGISTER_ELEMENT:
    GObject.type_register(OverlayCairo)
    __gstelementfactory__ = ("pyml_overlay_cairo", Gst.Rank.NONE, OverlayCairo)
else:
    Gst.warning(
        "The 'pyml_overlay_cairo' element will not be registered because a module is missing."
    )
