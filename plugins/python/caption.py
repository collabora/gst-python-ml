# Caption
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

CAN_REGISTER_ELEMENT = True
try:
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    gi.require_version("GstVideo", "1.0")
    gi.require_version("GLib", "2.0")
    gi.require_version("GstAnalytics", "1.0")

    from gi.repository import Gst, GObject, GstAnalytics, GLib  # noqa: E402
    import numpy as np
    import cv2
    from video_transform import VideoTransform
except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'pyml_caption' element will not be available. Error {e}"
    )


class Caption(VideoTransform):
    """
    GStreamer element for captioning video frames.
    """

    __gstmetadata__ = (
        "Caption",
        "Transform",
        "Captions video clips",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    __gsttemplates__ = (
        Gst.PadTemplate.new(
            "text_src",
            Gst.PadDirection.SRC,
            Gst.PadPresence.REQUEST,
            Gst.Caps.from_string("text/x-raw, format=utf8"),
        ),
    )

    prompt = GObject.Property(
        type=str,
        default="What is shown in this image?",
        nick="Custom Prompt",
        blurb="Custom prompt text for image analysis",
    )

    def __init__(self):
        super().__init__()
        self.model_name = "phi-3-vision"
        self.caption = "   "

    def do_set_property(self, property, value):
        if property.name == "prompt":
            self.prompt = value
            if self.engine:
                self.engine.prompt = value
        else:
            super().do_set_property(property, value)

    def do_get_property(self, property):
        if property.name == "prompt":
            return self.prompt
        else:
            return super().do_get_property(property)

    def do_start(self):
        # Create the `text_src` pad on start to link to downstream element if it exists
        self.text_src_pad = Gst.Pad.new_from_template(
            self.get_pad_template("text_src"), "text_src"
        )
        self.add_pad(self.text_src_pad)

        # Attempt to link text_src to downstream text_sink if available
        self.link_to_downstream_text_sink()
        return True

    def link_to_downstream_text_sink(self):
        """
        Attempts to find and link the `text_src` pad to a downstream `text_sink` pad.
        """
        self.logger.info("Attempting to link text_src pad to downstream text_sink pad")
        src_peer = self.get_static_pad("src").get_peer()
        if src_peer:
            downstream_element = src_peer.get_parent()
            text_sink_pad = downstream_element.get_static_pad("text_sink")
            if text_sink_pad:
                self.text_src_pad.link(text_sink_pad)
                self.logger.info("Successfully linked text_src to downstream text_sink")
            else:
                self.logger.warning(
                    "No text_sink pad found downstream to link with text_src"
                )
        else:
            self.logger.warning("No downstream peer found to link text_src")

    def push_text_buffer(self, text, buf_pts, buf_duration):
        """
        Pushes a text buffer to the `text_src` pad with proper timestamps.

        Args:
            text (str): The text to push as a buffer.
            buf_pts (int): The PTS of the associated video buffer.
            buf_duration (int): The duration of the associated video buffer.
        """
        text_buffer = Gst.Buffer.new_wrapped(text.encode("utf-8"))

        # Set the text buffer timestamps
        text_buffer.pts = buf_pts
        text_buffer.dts = buf_pts  # DTS is usually the same as PTS for text buffers
        # disable duration for now, as it freezes the pipeline
        # text_buffer.duration = buf_duration

        # Push the buffer
        ret = self.text_src_pad.push(text_buffer)
        if ret != Gst.FlowReturn.OK:
            self.logger.error(f"Failed to push text buffer: {ret}")

    def do_transform_ip(self, buf):
        """
        In-place transformation for object detection inference.
        """
        try:
            if self.get_model() is None:
                self.do_load_model()

            self.engine.prompt = self.prompt

            # Set a valid timestamp if none is set
            if buf.pts == Gst.CLOCK_TIME_NONE:
                buf.pts = Gst.util_uint64_scale(
                    Gst.util_get_timestamp(),
                    self.framerate_denom,
                    self.framerate_num * Gst.SECOND,
                )

            if buf.duration == Gst.CLOCK_TIME_NONE:
                buf.duration = Gst.SECOND // self.framerate_num

            # Map the input buffer to read the data
            with buf.map(Gst.MapFlags.READ | Gst.MapFlags.WRITE) as info:
                frame = np.ndarray(
                    shape=(self.height, self.width, 3),
                    dtype=np.uint8,
                    buffer=info.data,
                )
            # Check if rescaling is needed
            if (
                self.downsampled_width > 0
                and self.downsampled_width < self.width
                and self.downsampled_height > 0
                and self.downsampled_height < self.height
            ):
                # Perform rescaling using OpenCV
                resized_frame = cv2.resize(
                    frame,
                    (self.downsampled_width, self.downsampled_height),
                    interpolation=cv2.INTER_AREA,  # Best for shrinking images
                )

                # Replace the original frame's content with the resized one
                frame = resized_frame
                self.logger.info(
                    f"resized to dimensions {self.downsampled_width}, {self.downsampled_height}"
                )

            if self.engine:
                result = self.engine.forward(frame)
                if result:
                    self.caption = result
                    meta = GstAnalytics.buffer_add_analytics_relation_meta(buf)
                    if meta:
                        qk = GLib.quark_from_string(f"{result}")
                        ret, mtd = meta.add_one_cls_mtd(0, qk)
                        if ret:
                            self.logger.info(f"Successfully added caption {result}")
                        else:
                            self.logger.error("Failed to add classification metadata")
                    else:
                        self.logger.error(
                            "Failed to add GstAnalytics metadata to buffer"
                        )

            # Send text data through the `text_src` pad if it is linked
            if self.text_src_pad and self.text_src_pad.is_linked():
                self.push_text_buffer(self.caption, buf.pts, buf.duration)
            else:
                self.logger.warning(
                    "TextExtract: text_src pad is not linked, cannot push text buffer."
                )

            return Gst.FlowReturn.OK

        except Gst.MapError as e:
            self.logger.error(f"Mapping error: {e}")
            return Gst.FlowReturn.ERROR
        except Exception as e:
            self.logger.error(f"Error during transformation: {e}")
            return Gst.FlowReturn.ERROR


if CAN_REGISTER_ELEMENT:
    GObject.type_register(Caption)
    __gstelementfactory__ = ("pyml_caption", Gst.Rank.NONE, Caption)
else:
    GlobalLogger().warning(
        "The 'pyml_caption' element will not be registered because required modules are missing."
    )
