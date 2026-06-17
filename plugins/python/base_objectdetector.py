# object_detector_base.py
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

import traceback
from utils.runtime_utils import runtime_check_gstreamer_version
import gi
from video_transform import VideoTransform
from utils.format_converter import FormatConverter
from utils.muxed_buffer_processor import MuxedBufferProcessor  # Added import

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
gi.require_version("GstAnalytics", "1.0")
gi.require_version("GLib", "2.0")
from gi.repository import Gst, GstAnalytics, GObject, GLib  # noqa: E402
from utils.metadata import Metadata  # noqa: E402


class BaseObjectDetector(VideoTransform):
    """
    GStreamer element for object detection with batch processing support.
    Handles both single-frame buffers (no metadata) and batch buffers (metadata in last chunk).
    """

    def __init__(self):
        super().__init__()
        runtime_check_gstreamer_version()
        self.framerate_num = 30
        self.framerate_denom = 1
        self.format_converter = FormatConverter()
        self.metadata = Metadata("si")
        self.logger.info("Initialized BaseObjectDetector")
        self.__track = False
        self.__interval = 1
        self._det_counter = 0
        self._cached_results = None
        self._cached_num_sources = 1
        self._cached_id = None

    @GObject.Property(type=bool, default=False)
    def track(self):
        "Enable or disable tracking mode"
        if self.engine:
            return self.engine.track
        return self.__track

    @track.setter
    def track(self, value):
        self.__track = value
        if self.engine:
            self.engine.track = value

    @GObject.Property(type=int, default=1, minimum=1, maximum=10000)
    def interval(self):
        "Run detection every Nth frame and re-attach the previous detections on "
        "the frames in between (N=1 runs detection every frame). Lets downstream "
        "tracking/overlay stay per-frame while detection runs at a lower rate."
        return self.__interval

    @interval.setter
    def interval(self, value):
        self.__interval = max(1, int(value))

    def do_forward(self, frames):
        self.logger.info(
            f"Forward called with frames shape: {frames.shape if frames is not None else 'None'}"
        )
        if self.engine:
            self.engine.track = self.track
            result = self.engine.do_forward(frames)
            self.logger.debug(f"Forward result: {result} (type: {type(result)})")
            return result
        return None

    def do_transform_ip(self, buf):
        """
        Transform the input buffer using MuxedBufferProcessor for frame extraction.
        """
        self.logger.info(f"Transforming buffer: {hex(id(buf))}")
        try:
            run_detect = (self._det_counter % self.__interval) == 0
            self._det_counter += 1

            if run_detect:
                # Use MuxedBufferProcessor to extract frames and metadata
                muxed_processor = MuxedBufferProcessor(
                    self.logger,
                    self.width,
                    self.height,
                    self.framerate_num,
                    self.framerate_denom,
                )
                frames, id_str, num_sources, format = muxed_processor.extract_frames(
                    buf, self.sinkpad
                )
                if frames is None:
                    self.logger.error("Failed to extract frames")
                    return Gst.FlowReturn.ERROR

                results = self.do_forward(frames)
                if results is None:
                    self.logger.error("Inference returned None")
                    return Gst.FlowReturn.ERROR

                self._cached_results = results
                self._cached_num_sources = num_sources
                self._decode_results(buf, results, num_sources)
            elif self._cached_results is not None:
                # Skip inference on this frame and re-attach the previous
                # detections so downstream tracking/overlay stay per-frame.
                self._decode_results(
                    buf, self._cached_results, self._cached_num_sources
                )

            attached_meta = GstAnalytics.buffer_get_analytics_relation_meta(buf)
            if attached_meta:
                count = GstAnalytics.relation_get_length(attached_meta)
                self.logger.info(f"Total metadata relations attached: {count}")
            else:
                self.logger.debug("No detections on this buffer")

            return Gst.FlowReturn.OK

        except Exception as e:
            self.logger.error(f"Transform error: {e}\n{traceback.format_exc()}")
            return Gst.FlowReturn.ERROR

    def _decode_results(self, buf, results, num_sources):
        if num_sources == 1:
            self.do_decode(buf, results, stream_idx=0)
        else:
            results_list = results if isinstance(results, list) else [results]
            if len(results_list) != num_sources:
                self.logger.error(
                    f"Expected {num_sources} results, got {len(results_list)}"
                )
                return
            for idx, result in enumerate(results_list):
                if result is None:
                    self.logger.warning(f"Frame {idx} result is None")
                    continue
                self.do_decode(buf, result, stream_idx=idx)

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
