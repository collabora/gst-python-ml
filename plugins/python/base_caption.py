# BaseCaption
# Copyright (C) 2024-2026 Collabora Ltd.
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


import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
gi.require_version("GstVideo", "1.0")

from gi.repository import Gst, GObject, GstBase  # noqa: E402
import backend  # noqa: E402
from backend import analytics, frameio  # noqa: E402
from video_transform import VideoTransform

# Building a Gst object needs Gst.init, which only the gst backend calls.
if backend.BACKEND == "gst":
    TEXT_CAPS = Gst.Caps.from_string("text/x-raw, format=utf8")


class BaseCaption(VideoTransform):
    """
    Base GStreamer element for captioning video frames.
    """

    __gstmetadata__ = (
        "BaseCaption",
        "Transform",
        "Captions video clips",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    if backend.BACKEND == "gst":
        __gsttemplates__ = (
            Gst.PadTemplate.new(
                "text_src", Gst.PadDirection.SRC, Gst.PadPresence.REQUEST, TEXT_CAPS
            ),
        )

    def __init__(self):
        super().__init__()
        self._prompt = "What is shown in this image?"
        self.text_src_pad = None

    @GObject.Property(type=str)
    def system_prompt(self):
        "Custom system prompt text"
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value):
        self._system_prompt = value

    @GObject.Property(type=str)
    def prompt(self):
        "Custom prompt text"
        return self._prompt

    @prompt.setter
    def prompt(self, value):
        self._prompt = value

    # make read only
    @GObject.Property(type=str)
    def engine_name(self):
        "Machine Learning Engine to use : pytorch, tflite, tensorflow, onnx or openvino, or custom engine name"
        return self.mgr.engine_name

    @engine_name.setter
    def engine_name(self, value):
        raise ValueError(
            "The 'engine_name' property cannot be set in this derived class."
        )

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

    def push_text_buffer(self, text, buf_pts, buf_dts, buf_duration):
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
        text_buffer.dts = buf_dts
        # Put a long duration so the subtitles are visible
        text_buffer.duration = 60 * Gst.SECOND

        # Push the buffer
        ret = self.text_src_pad.push(text_buffer)
        if ret != Gst.FlowReturn.OK:
            self.logger.warning(f"Failed to push text buffer: {ret}")

    def do_transform_ip(self, buf):
        """
        In-place transformation for captioning inference, extracting frame(s)
        through the backend frame I/O.
        """
        try:
            # Extract frame(s) through the backend's frame I/O.
            frames, num_sources, _ = frameio.read_frames(
                buf, self.sinkpad, self.width, self.height
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
                if self.engine:
                    result = self.engine.do_forward(frame)
                    if result:
                        self.caption = result
                        meta = analytics.add_relation_meta(buf)
                        if meta:
                            mtd = analytics.add_classification(meta, 0, f"{result}")
                            if mtd is not None:
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
                            self.push_text_buffer(
                                self.caption, buf.pts, buf.dts, buf.duration
                            )
                        else:
                            self.logger.warning(
                                "TextExtract: text_src pad is not linked, cannot push text buffer."
                            )
            else:
                # Batch case
                self.logger.info(f"Processing batch with num_sources={num_sources}")
                if self.engine:
                    results = self.engine.do_forward(frames)
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
                            meta = analytics.add_relation_meta(buf)
                            if meta:
                                mtd = analytics.add_classification(
                                    meta, idx, f"stream_{idx}_{result}"
                                )
                                if mtd is not None:
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
            text_event = (
                Gst.Event.new_caps(TEXT_CAPS)
                if event.type == Gst.EventType.CAPS
                else event
            )
            self.text_src_pad.push_event(text_event)
        return GstBase.BaseTransform.do_sink_event(self, event)
