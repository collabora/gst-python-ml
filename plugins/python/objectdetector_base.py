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
    GStreamer element for object detection.
    Batch handling with clean logs. Marker: WORKING_2025_03_04_V7.
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
        self.logger.info("Initialized ObjectDetectorBase - WORKING_2025_03_04_V7")

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

    def forward(self, frame):
        self.logger.info(
            f"Forward called with frame shape: {frame.shape if frame is not None else 'None'}"
        )
        if self.engine:
            self.engine.track = self.track
            result = self.engine.forward(frame)
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

            if num_chunks > 1:
                self.logger.info(f"Processing batch with {num_chunks} chunks")
                id_str, num_sources = self.metadata.read(buf)
                self.logger.info(f"Metadata: ID={id_str}, num_sources={num_sources}")

                batch_results = []
                for i in range(num_chunks - 1):
                    with buf.peek_memory(i).map(Gst.MapFlags.READ) as info:
                        frame = self.format_converter.get_rgb_frame(
                            info, format, self.height, self.width
                        )
                        result = self.forward(frame)
                        self.logger.debug(
                            f"Frame {i} result: {result} (type: {type(result)})"
                        )
                        batch_results.append(result)

                if len(batch_results) != num_sources:
                    self.logger.error(
                        f"Expected {num_sources} results, got {len(batch_results)}"
                    )
                    return Gst.FlowReturn.ERROR

                for idx, result in enumerate(batch_results):
                    if result is None:
                        self.logger.warning(f"Frame {idx} result is None")
                        continue
                    self.logger.debug(f"Attaching metadata for frame {idx}")
                    result_obj = (
                        result[0]
                        if isinstance(result, list) and len(result) > 0
                        else result
                    )
                    self.logger.debug(
                        f"Before do_decode: {result_obj} (type: {type(result_obj)})"
                    )
                    # Pass stream index to do_decode
                    self.do_decode(buf, result_obj, stream_idx=idx)

                attached_meta = GstAnalytics.buffer_get_analytics_relation_meta(buf)
                if attached_meta:
                    count = GstAnalytics.relation_get_length(attached_meta)
                    self.logger.info(f"Total metadata relations attached: {count}")
                else:
                    self.logger.error("No metadata attached to buffer")

            else:
                self.logger.info("Single frame mode")
                with buf.map(Gst.MapFlags.READ | Gst.MapFlags.WRITE) as info:
                    if info.data is None:
                        self.logger.error("Buffer map failed")
                        return Gst.FlowReturn.ERROR
                    frame = self.format_converter.get_rgb_frame(
                        info, format, self.height, self.width
                    )
                    if frame is None or not isinstance(frame, np.ndarray):
                        self.logger.error("Invalid frame")
                        return Gst.FlowReturn.ERROR
                    results = self.forward(frame)
                    self.logger.debug(
                        f"Single frame result: {results} (type: {type(results)})"
                    )
                    if isinstance(results, dict):
                        self.do_decode(buf, results, stream_idx=0)
                    elif isinstance(results, list):
                        for i, result in enumerate(results):
                            if result is None:
                                self.logger.warning(f"Result {i} is None")
                                continue
                            result_obj = (
                                result if not isinstance(result, list) else result[0]
                            )
                            self.logger.debug(
                                f"Before do_decode: {result_obj} (type: {type(result_obj)})"
                            )
                            self.do_decode(buf, result_obj, stream_idx=0)
                    else:
                        self.logger.error(f"Unexpected type: {type(results)}")
                        return Gst.FlowReturn.ERROR

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
        elif hasattr(output, "boxes"):  # Direct Results object
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
