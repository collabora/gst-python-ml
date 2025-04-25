# llm_stream_filter.py
# Copyright (C) 2024-2025 Collabora Ltd.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Library General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
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
    from gi.repository import Gst, GObject, GstAnalytics, GLib
    import numpy as np
    import cv2

    from global_logger import GlobalLogger
    from muxed_buffer_processor import MuxedBufferProcessor
    from video_transform import VideoTransform

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'pyml_llmstreamfilter' element will not be available. Error {e}"
    )


class LLMStreamFilter(VideoTransform):
    """
    GStreamer element that captions video frames, processes captions with an LLM to select
    the N most interesting ones, and outputs only those streams. Supports dynamic updates
    to the prompt and number of streams during runtime.
    """

    __gstmetadata__ = (
        "LLMStreamFilter",
        "Transform",
        "Captions video clips and selects N most interesting streams",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    __gsttemplates__ = (
        Gst.PadTemplate.new(
            "video_src",
            Gst.PadDirection.SRC,
            Gst.PadPresence.ALWAYS,
            Gst.Caps.from_string("video/x-raw"),
        ),
        Gst.PadTemplate.new(
            "text_src",
            Gst.PadDirection.SRC,
            Gst.PadPresence.REQUEST,
            Gst.Caps.from_string("text/x-raw, format=utf8"),
        ),
    )

    num_streams = GObject.Property(
        type=int,
        default=2,
        nick="Number of Streams",
        blurb="Number of streams to select (N)",
    )

    prompt = GObject.Property(
        type=str,
        default="Choose the {n} most interesting captions from the following list:\n{captions}",
        nick="LLM Prompt",
        blurb="Prompt for selecting the most interesting captions",
    )

    def __init__(self):
        super().__init__()
        self.model_name = "phi-3.5-vision"  # Caption model
        self.llm_model_name = "llama-3.1"  # LLM model, adjust as needed
        self.caption_engine = None
        self.llm_engine = None
        self.selected_streams = []
        self.logger = GlobalLogger()

    def do_set_property(self, prop, value):
        """
        Handle property changes, including runtime updates to prompt and num_streams.
        """
        if prop.name == "num-streams":
            self.num_streams = value
            self.logger.info(f"Updated num_streams to {value}")
        elif prop.name == "prompt":
            self.prompt = value
            if self.llm_engine:
                self.llm_engine.prompt = value  # Update LLM engine prompt if available
            self.logger.info(f"Updated prompt to: {value}")
        else:
            super().do_set_property(prop, value)

    def do_get_property(self, prop):
        """
        Retrieve property values.
        """
        if prop.name == "num-streams":
            return self.num_streams
        elif prop.name == "prompt":
            return self.prompt
        else:
            return super().do_get_property(prop)

    def do_start(self):
        """
        Initialize the element, including pads and engines.
        """
        # Create the text_src pad
        self.text_src_pad = Gst.Pad.new_from_template(
            self.get_pad_template("text_src"), "text_src"
        )
        self.add_pad(self.text_src_pad)

        # Initialize caption and LLM engines
        self.do_load_model()
        if not self.caption_engine or not self.llm_engine:
            self.logger.error("Failed to initialize caption or LLM engine")
            return False

        self.link_to_downstream_text_sink()
        return True

    def do_load_model(self):
        """
        Load caption and LLM models.
        """
        try:
            if not self.caption_engine:
                self.caption_engine = self.get_engine(self.model_name)
                if not self.caption_engine:
                    self.logger.error("Failed to load caption engine")
                    return False
            if not self.llm_engine:
                self.llm_engine = self.get_engine(self.llm_model_name)
                if not self.llm_engine:
                    self.logger.error("Failed to load LLM engine")
                    return False
            self.llm_engine.prompt = (
                self.prompt
            )  # Ensure LLM engine uses current prompt
            return True
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            return False

    def link_to_downstream_text_sink(self):
        """
        Attempts to link the text_src pad to a downstream text_sink pad.
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
                self.logger.warning("No text_sink pad found downstream")
        else:
            self.logger.warning("No downstream peer found")

    def push_text_buffer(self, text, buf_pts, buf_duration):
        """
        Pushes a text buffer to the text_src pad.
        """
        text_buffer = Gst.Buffer.new_wrapped(text.encode("utf-8"))
        text_buffer.pts = buf_pts
        text_buffer.dts = buf_pts
        # text_buffer.duration = buf_duration  # Disabled to avoid pipeline freeze
        ret = self.text_src_pad.push(text_buffer)
        if ret != Gst.FlowReturn.OK:
            self.logger.error(f"Failed to push text buffer: {ret}")

    def select_interesting_streams(self, captions, num_streams):
        """
        Uses the LLM to select the N most interesting captions.
        Returns the indices of the selected streams.
        """
        try:
            captions_text = "\n".join([f"{i}: {c}" for i, c in enumerate(captions)])
            prompt = self.prompt.format(n=num_streams, captions=captions_text)
            self.logger.info(f"LLM prompt: {prompt}")

            generated_text = self.llm_engine.generate(prompt)
            self.logger.info(f"LLM output: {generated_text}")

            # Parse LLM output (assuming it returns indices or captions)
            selected_indices = []
            for line in generated_text.split("\n"):
                try:
                    idx = int(line.split(":")[0])
                    if 0 <= idx < len(captions):
                        selected_indices.append(idx)
                except (ValueError, IndexError):
                    continue

            return selected_indices[:num_streams]
        except Exception as e:
            self.logger.error(f"Error in LLM processing: {e}")
            return []

    def do_transform_ip(self, buf):
        """
        In-place transformation: captions frames, selects N streams, and outputs results.
        """
        try:
            if not self.caption_engine or not self.llm_engine:
                self.do_load_model()
                if not self.caption_engine or not self.llm_engine:
                    return Gst.FlowReturn.ERROR

            # Initialize MuxedBufferProcessor
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
                    Gst.util_get_timestamp(), 1, 30 * Gst.SECOND
                )
            if buf.duration == Gst.CLOCK_TIME_NONE:
                buf.duration = Gst.SECOND // 30

            captions = []
            if num_sources == 1:
                # Single-frame case
                frame = frames
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
                        f"Resized to {self.downsampled_width}x{self.downsampled_height}"
                    )

                result = self.caption_engine.forward(frame)
                captions.append(result if result else "")
            else:
                # Batch case
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
                        f"Resized batch to {self.downsampled_width}x{self.downsampled_height}"
                    )

                results = self.caption_engine.forward(frames)
                results_list = (
                    results if isinstance(results, list) else [results] * num_sources
                )
                if len(results_list) != num_sources:
                    self.logger.error(
                        f"Expected {num_sources} results, got {len(results_list)}"
                    )
                    return Gst.FlowReturn.ERROR
                captions = [r if r else "" for r in results_list]

            # Select the most interesting streams
            self.selected_streams = self.select_interesting_streams(
                captions, self.num_streams
            )
            self.logger.info(f"Selected streams: {self.selected_streams}")

            # Add metadata and push text buffers for selected streams
            for idx, caption in enumerate(captions):
                if idx in self.selected_streams:
                    meta = GstAnalytics.buffer_add_analytics_relation_meta(buf)
                    if meta:
                        qk = GLib.quark_from_string(f"stream_{idx}_{caption}")
                        ret, mtd = meta.add_one_cls_mtd(idx, qk)
                        if ret:
                            self.logger.info(f"Stream {idx}: Added caption {caption}")
                        else:
                            self.logger.error(f"Stream {idx}: Failed to add metadata")
                    else:
                        self.logger.error(
                            f"Stream {idx}: Failed to add GstAnalytics metadata"
                        )

                    if self.text_src_pad and self.text_src_pad.is_linked():
                        frame_pts = buf.pts + (idx * (buf.duration // num_sources))
                        self.push_text_buffer(
                            caption, frame_pts, buf.duration // num_sources
                        )
                    else:
                        self.logger.warning(f"Stream {idx}: text_src pad not linked")

            return Gst.FlowReturn.OK

        except Exception as e:
            self.logger.error(f"Error during transformation: {e}")
            return Gst.FlowReturn.ERROR


if CAN_REGISTER_ELEMENT:
    GObject.type_register(LLMStreamFilter)
    __gstelementfactory__ = ("pyml_llmstreamfilter", Gst.Rank.NONE, LLMStreamFilter)
else:
    GlobalLogger().warning(
        "The 'pyml_llmstreamfilter' element will not be registered because required modules are missing."
    )
