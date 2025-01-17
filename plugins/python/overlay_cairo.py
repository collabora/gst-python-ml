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
        nick="Metadata File Path",
        blurb="Path to the JSON file containing frame metadata with bounding boxes and tracking data",
        flags=GObject.ParamFlags.READWRITE,
    )

    # Add the tracking property
    tracking = GObject.Property(
        type=bool,
        default=False,
        nick="Enable Tracking Display",
        blurb="Enable or disable tracking display",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super(OverlayCairo, self).__init__()
        self.meta_path = None
        self.tracking = True
        self.preloaded_metadata = {}  # Dictionary to store frame-indexed metadata
        self.frame_counter = 0
        self.width = 640
        self.height = 480
        self.history = []
        self.max_history_length = 5000
        # Color palette for tracking IDs
        self.color_palette = [
            (1.0, 0.0, 0.0),  # Red
            (0.0, 1.0, 0.0),  # Green
            (0.0, 0.0, 1.0),  # Blue
            (1.0, 1.0, 0.0),  # Yellow
            (1.0, 0.0, 1.0),  # Magenta
            (0.0, 1.0, 1.0),  # Cyan
        ]
        # Dictionary to store ID-to-color mapping
        self.id_color_map = {}

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
            Gst.info(f"Tracking set to: {self.tracking}")
        else:
            raise AttributeError(f"Unknown property {prop.name}")

    def on_message(self, bus, message):
        if message.type == Gst.MessageType.EOS:
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

            # Draw the fading history
            if self.tracking:
                for point in self.history:
                    self.draw_tracking_box(
                        cr,
                        point["center"],
                        point["color"],
                        point["opacity"],
                    )
                    # Reduce the opacity for fading effect
                    point["opacity"] *= 0.9

            # Draw bounding boxes, labels, and new tracking boxes
            for data in metadata:
                self.draw_bounding_box_with_cairo(cr, data["box"])
                self.draw_label_with_cairo(
                    cr, data["label"], data["box"]["x1"], data["box"]["y1"]
                )

                if self.tracking:
                    track_id = data.get("track_id")
                    if track_id is not None:
                        color = self.get_color_for_id(track_id)
                        center = {
                            "x": (data["box"]["x1"] + data["box"]["x2"]) / 2,
                            "y": (data["box"]["y1"] + data["box"]["y2"]) / 2,
                        }
                        self.draw_tracking_box(cr, center, color, 1.0)

                        # Add new tracking box to history
                        self.history.append(
                            {"center": center, "color": color, "opacity": 1.0}
                        )

            # Trim history if it exceeds max length
            if self.tracking:
                self.history = [
                    point for point in self.history if point["opacity"] > 0.1
                ]

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

    def get_color_for_id(self, track_id):
        """Get a color for the given track ID."""
        if track_id not in self.id_color_map:
            # Assign a color based on the track ID, cycling through the palette
            color_index = len(self.id_color_map) % len(self.color_palette)
            self.id_color_map[track_id] = self.color_palette[color_index]
        return self.id_color_map[track_id]

    def draw_tracking_box(self, cr, center, color, opacity):
        """Draw a small box for tracking at the given center point with fading effect."""
        size = 10  # Size of the tracking box
        half_size = size // 2

        # Set the color with opacity
        cr.set_source_rgba(color[0], color[1], color[2], opacity)

        # Draw the filled box
        cr.rectangle(center["x"] - half_size, center["y"] - half_size, size, size)
        cr.fill()

    def draw_bounding_box_with_cairo(self, cr, box):
        """Draw a bounding box using Cairo."""
        cr.set_line_width(2.0)
        cr.set_source_rgb(1, 0, 0)  # Red color for bounding box
        cr.rectangle(box["x1"], box["y1"], box["x2"] - box["x1"], box["y2"] - box["y1"])
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
