# caption.py
# Copyright (C) 2024-2025 Collabora Ltd.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Library General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
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
from muxed_buffer_processor import MuxedBufferProcessor  # Added import

CAN_REGISTER_ELEMENT = True
try:
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    gi.require_version("GstVideo", "1.0")
    gi.require_version("GLib", "2.0")
    gi.require_version("GstAnalytics", "1.0")

    from gi.repository import Gst, GObject, GstAnalytics, GLib, GstBase  # noqa: E402
    import numpy as np
    import cv2
    from video_transform import VideoTransform
except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'pyml_caption' element will not be available. Error {e}"
    )

TEXT_CAPS = Gst.Caps.from_string("text/x-raw, format=utf8")

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
            TEXT_CAPS
        ),
    )

    @GObject.Property(type=str)
    def prompt(self):
        "Custom prompt text for image analysis"
        return self.__prompt

    @prompt.setter
    def prompt(self, value):
        self.__prompt = value
        if self.engine:
            self.engine.prompt = value

    def __init__(self):
        super().__init__()
        self.model_name = "phi-3.5-vision"
        self.caption = "   "
        self.text_src_pad = None

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

    def do_request_new_pad(self, template, name, caps):
        if self.text_src_pad:
            self.logger.error("Element already has a text_src")
            return None
        if name != template.name_template:
            self.logger.error("Invalid pad name")
            return None

        self.text_src_pad = Gst.Pad.new_from_template(template, name)
        self.add_pad(self.text_src_pad)
        self.text_src_pad.set_active(True)

        return self.text_src_pad

    def do_release_pad(self, pad):
        self.remove_pad(pad)
        pad.set_active(False)
        self.text_src_pad = None

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
            self.logger.warning(f"Failed to push text buffer: {ret}")

    def do_transform_ip(self, buf):
        """
        In-place transformation for captioning inference using MuxedBufferProcessor.
        """
        try:
            if self.get_model() is None:
                self.do_load_model()

            self.engine.prompt = self.__prompt

            # Initialize MuxedBufferProcessor with default framerate
            muxed_processor = MuxedBufferProcessor(
                self.logger,
                self.width,
                self.height,
                framerate_num=30,
                framerate_denom=1,
            )
            frames, id_str, num_sources, format = muxed_processor.extract_frames(
                buf, self.sinkpad
            )
            if frames is None:
                self.logger.error("Failed to extract frames")
                return Gst.FlowReturn.ERROR

            # Set timestamps if none are set
            if buf.pts == Gst.CLOCK_TIME_NONE:
                buf.pts = Gst.util_uint64_scale(
                    Gst.util_get_timestamp(),
                    1,  # framerate_denom
                    30 * Gst.SECOND,  # framerate_num
                )
            if buf.duration == Gst.CLOCK_TIME_NONE:
                buf.duration = Gst.SECOND // 30  # framerate_num

            # Process frames (single or batch)
            if num_sources == 1:
                # Single-frame case
                frame = frames
                # Check if rescaling is needed
                if (
                    self.downsampled_width > 0
                    and self.downsampled_width < self.width
                    and self.downsampled_height > 0
                    and self.downsampled_height < self.height
                ):
                    frame = cv2.resize(
                        frame,
                        (self.downsampled_width, self.downsampled_height),
                        interpolation=cv2.INTER_AREA,
                    )
                    self.logger.info(
                        f"Resized to dimensions {self.downsampled_width}, {self.downsampled_height}"
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
                                self.logger.error(
                                    "Failed to add classification metadata"
                                )
                        else:
                            self.logger.error(
                                "Failed to add GstAnalytics metadata to buffer"
                            )

                        # Push text buffer if text_src pad is linked
                        if self.text_src_pad:
                            self.push_text_buffer(self.caption, buf.pts, buf.duration)
                        else:
                            self.logger.warning(
                                "TextExtract: text_src pad is not linked, cannot push text buffer."
                            )
            else:
                # Batch case
                self.logger.info(
                    f"Processing batch with ID={id_str}, num_sources={num_sources}"
                )
                # Rescale frames if needed
                if (
                    self.downsampled_width > 0
                    and self.downsampled_width < self.width
                    and self.downsampled_height > 0
                    and self.downsampled_height < self.height
                ):
                    frames = np.stack(
                        [
                            cv2.resize(
                                frame,
                                (self.downsampled_width, self.downsampled_height),
                                interpolation=cv2.INTER_AREA,
                            )
                            for frame in frames
                        ],
                        axis=0,
                    )
                    self.logger.info(
                        f"Resized batch to dimensions {self.downsampled_width}, {self.downsampled_height}"
                    )

                if self.engine:
                    results = self.engine.forward(frames)
                    if results is None:
                        self.logger.error("Inference returned None")
                        return Gst.FlowReturn.ERROR

                    # Ensure results is a list for batch processing
                    results_list = (
                        results
                        if isinstance(results, list)
                        else [results] * num_sources
                    )
                    if len(results_list) != num_sources:
                        self.logger.error(
                            f"Expected {num_sources} results, got {len(results_list)}"
                        )
                        return Gst.FlowReturn.ERROR

                    for idx, result in enumerate(results_list):
                        if result:
                            caption = result
                            meta = GstAnalytics.buffer_add_analytics_relation_meta(buf)
                            if meta:
                                qk = GLib.quark_from_string(f"stream_{idx}_{result}")
                                ret, mtd = meta.add_one_cls_mtd(idx, qk)
                                if ret:
                                    self.logger.info(
                                        f"Stream {idx}: Successfully added caption {result}"
                                    )
                                else:
                                    self.logger.error(
                                        f"Stream {idx}: Failed to add classification metadata"
                                    )
                            else:
                                self.logger.error(
                                    f"Stream {idx}: Failed to add GstAnalytics metadata"
                                )

                            # Push text buffer for each frame
                            if self.text_src_pad:
                                # Adjust PTS for each frame in the batch
                                frame_pts = buf.pts + (
                                    idx * (buf.duration // num_sources)
                                )
                                self.push_text_buffer(
                                    caption, frame_pts, buf.duration // num_sources
                                )
                            else:
                                self.logger.warning(
                                    f"Stream {idx}: TextExtract: text_src pad is not linked, cannot push text buffer."
                                )
                        else:
                            self.logger.warning(f"Stream {idx}: No caption generated")

            return Gst.FlowReturn.OK

        except Exception as e:
            self.logger.error(f"Error during transformation: {e}")
            return Gst.FlowReturn.ERROR

    def do_sink_event(self, event):
        if self.text_src_pad:
            text_event = Gst.Event.new_caps(TEXT_CAPS) if event.type == Gst.EventType.CAPS else event
            self.text_src_pad.push_event(text_event)
        return GstBase.BaseTransform.do_sink_event(self, event)


if CAN_REGISTER_ELEMENT:
    GObject.type_register(Caption, "pyml_caption")
    __gstelementfactory__ = ("pyml_caption", Gst.Rank.NONE, Caption)
else:
    GlobalLogger().warning(
        "The 'pyml_caption' element will not be registered because required modules are missing."
    )
