# ObjectDetectorBase
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

from utils import runtime_check_gstreamer_version
import gi
import numpy as np
from video_transform import VideoTransform
from format_converter import FormatConverter

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
gi.require_version("GstAnalytics", "1.0")
gi.require_version("GLib", "2.0")
from gi.repository import Gst, GstAnalytics, GObject, GLib  # noqa: E402
from metadata import Metadata  # noqa: E402


class ObjectDetectorBase(VideoTransform):
    """
    GStreamer element for object detection with batch processing support.
    Handles both single-frame buffers (no metadata) and batch buffers (metadata in last chunk).
    Marker: WORKING_2025_03_11_BATCH_V3.
    """

    track = GObject.Property(
        type=bool,
        default=False,
        nick="Track Mode",
        blurb="Enable or disable tracking mode",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        runtime_check_gstreamer_version()
        self.framerate_num = 30
        self.framerate_denom = 1
        self.format_converter = FormatConverter()
        self.metadata = Metadata("si")
        self.logger.info("Initialized ObjectDetectorBase - WORKING_2025_03_11_BATCH_V3")

    def do_set_property(self, prop, value):
        self.logger.info(f"Setting property {prop.name} to {value}")
        if prop.name == "track":
            self.track = value
            if self.engine:
                self.engine.track = value
        else:
            raise AttributeError(f"Unknown property {prop.name}")

    def do_get_property(self, prop):
        self.logger.info(f"Getting property {prop.name}")
        if prop.name == "track":
            if self.engine:
                return self.engine.track
            return self.track
        else:
            raise AttributeError(f"Unknown property {prop.name}")

    def forward(self, frames):
        self.logger.info(
            f"Forward called with frames shape: {frames.shape if frames is not None else 'None'}"
        )
        if self.engine:
            self.engine.track = self.track
            result = self.engine.forward(frames)
            self.logger.debug(f"Forward result: {result} (type: {type(result)})")
            return result
        return None

    def do_transform_ip(self, buf):
        self.logger.info(f"Transforming buffer: {hex(id(buf))}")
        try:
            if self.get_model() is None:
                self.logger.info("Loading model")
                self.do_load_model()

            if buf.pts == Gst.CLOCK_TIME_NONE:
                buf.pts = Gst.util_uint64_scale(
                    Gst.util_get_timestamp(),
                    self.framerate_denom,
                    self.framerate_num * Gst.SECOND,
                )

            num_chunks = buf.n_memory()
            format = self.format_converter.get_video_format(buf, self.sinkpad)
            self.logger.info(f"Chunks: {num_chunks}, format: {format}")

            if num_chunks < 1:
                self.logger.error("Buffer has no memory chunks")
                return Gst.FlowReturn.ERROR

            # Single frame case: no metadata
            if num_chunks == 1:
                self.logger.info("Single frame mode (no metadata)")
                with buf.peek_memory(0).map(Gst.MapFlags.READ) as info:
                    frame = self.format_converter.get_rgb_frame(
                        info, format, self.height, self.width
                    )
                    if frame is None or not isinstance(frame, np.ndarray):
                        self.logger.error("Invalid frame")
                        return Gst.FlowReturn.ERROR
                    results = self.forward(frame)
                    if results is None:
                        self.logger.error("Inference returned None")
                        return Gst.FlowReturn.ERROR
                    self.do_decode(buf, results, stream_idx=0)

            # Batch case: last chunk is metadata
            else:
                self.logger.info(f"Batch mode with {num_chunks} chunks")
                num_frames = num_chunks - 1
                frames = []
                for i in range(num_frames):
                    with buf.peek_memory(i).map(Gst.MapFlags.READ) as info:
                        frame = self.format_converter.get_rgb_frame(
                            info, format, self.height, self.width
                        )
                        if frame is None or not isinstance(frame, np.ndarray):
                            self.logger.error(f"Invalid frame at index {i}")
                            return Gst.FlowReturn.ERROR
                        frames.append(frame)

                # Read metadata from the last chunk
                id_str, num_sources = self.metadata.read(buf)
                self.logger.info(f"Metadata: ID={id_str}, num_sources={num_sources}")
                if num_sources != num_frames:
                    self.logger.error(
                        f"Metadata num_sources ({num_sources}) does not match frame count ({num_frames})"
                    )
                    return Gst.FlowReturn.ERROR

                batch_frames = np.stack(frames, axis=0)
                self.logger.info(f"Processing batch with shape: {batch_frames.shape}")
                results = self.forward(batch_frames)
                if results is None:
                    self.logger.error("Inference returned None")
                    return Gst.FlowReturn.ERROR

                results_list = results if isinstance(results, list) else [results]
                if len(results_list) != num_frames:
                    self.logger.error(
                        f"Expected {num_frames} results, got {len(results_list)}"
                    )
                    return Gst.FlowReturn.ERROR

                for idx, result in enumerate(results_list):
                    if result is None:
                        self.logger.warning(f"Frame {idx} result is None")
                        continue
                    self.do_decode(buf, result, stream_idx=idx)

            attached_meta = GstAnalytics.buffer_get_analytics_relation_meta(buf)
            if attached_meta:
                count = GstAnalytics.relation_get_length(attached_meta)
                self.logger.info(f"Total metadata relations attached: {count}")
            else:
                self.logger.error("No metadata attached to buffer")

            return Gst.FlowReturn.OK

        except Exception as e:
            self.logger.error(f"Transform error: {e}")
            return Gst.FlowReturn.ERROR

    def do_decode(self, buf, output, stream_idx=0):
        self.logger.info(
            f"Decoding for stream {stream_idx}: {output} (type: {type(output)})"
        )
        if isinstance(output, dict):
            self.logger.info(f"Stream {stream_idx} - Processing dict")
            boxes = output["boxes"]
            labels = output["labels"]
            scores = output["scores"]
        elif hasattr(output, "boxes"):  # Direct Results object (e.g., Ultralytics YOLO)
            self.logger.info(f"Stream {stream_idx} - Processing Ultralytics Results")
            boxes = output.boxes.xyxy.cpu().numpy()  # [N, 4]
            scores = output.boxes.conf.cpu().numpy()  # [N]
            labels = output.boxes.cls.cpu().numpy().astype(int)  # [N]
        elif (
            isinstance(output, list) and len(output) >= 6
        ):  # [x1, y1, x2, y2, score, label]
            self.logger.info(f"Stream {stream_idx} - Processing list of detections")
            boxes = [[det[0], det[1], det[2], det[3]] for det in output]
            scores = [det[4] for det in output]
            labels = [int(det[5]) for det in output]
        else:
            self.logger.error(
                f"Stream {stream_idx} - Unrecognized format: {output} (type: {type(output)})"
            )
            return

        meta = GstAnalytics.buffer_add_analytics_relation_meta(buf)
        if not meta:
            self.logger.error(
                f"Stream {stream_idx} - Failed to add analytics relation metadata"
            )
            return

        self.logger.info(f"Stream {stream_idx} - Adding {len(boxes)} detections")
        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            x1, y1, x2, y2 = box
            qk_string = f"stream_{stream_idx}_label_{label}"
            qk = GLib.quark_from_string(qk_string)
            ret, od_mtd = meta.add_od_mtd(qk, x1, y1, x2 - x1, y2 - y1, score)
            if not ret:
                self.logger.error(
                    f"Stream {stream_idx} - Failed to add od_mtd for detection {i}"
                )
                continue
            self.logger.info(
                f"Stream {stream_idx} - Added detection {i}: label={qk_string}, x1={x1}, y1={y1}, w={x2-x1}, h={y2-y1}, score={score}"
            )

        attached_meta = GstAnalytics.buffer_get_analytics_relation_meta(buf)
        if attached_meta:
            count = GstAnalytics.relation_get_length(attached_meta)
            self.logger.info(
                f"Stream {stream_idx} - Metadata relations after adding: {count}"
            )
