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

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
gi.require_version("GstVideo", "1.0")
gi.require_version("GstAnalytics", "1.0")
gi.require_version("GLib", "2.0")
from gi.repository import Gst, GstAnalytics, GObject, GLib  # noqa: E402


class ObjectDetectorBase(VideoTransform):
    """
    GStreamer element for object detection with a machine learning model.
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

    def do_set_property(self, prop, value):
        """Set the properties of the object."""
        if prop.name == "track":
            self.track = value
            if self.engine:
                self.engine.track = value  # Set the track flag on the engine_name
        else:
            raise AttributeError(f"Unknown property {prop.name}")

    def do_get_property(self, prop):
        """Get the properties of the object."""
        if prop.name == "track":
            if self.engine:
                return self.engine.track  # Get the track flag from the engine_name
            return self.track
        else:
            raise AttributeError(f"Unknown property {prop.name}")

    def forward(self, frame):
        if self.engine:
            self.engine.track = self.track
            return self.engine.forward(frame)
        else:
            return None

    def _extract_rgb(self, data: np.ndarray, format: str) -> np.ndarray:
        """
        Extracts the RGB channels from an image in ABGR, BGRA, RGBA, RGB, or BGR format.

        Parameters:
            data (np.ndarray): The input image data with either three or four channels,
                            with shape (height, width, 3) or (height, width, 4).
            format (str): The format of the input data. Expected values are 'ABGR', 'BGRA', 'RGBA', 'RGB', or 'BGR'.

        Returns:
            np.ndarray: A new image array with only the RGB channels, shape (height, width, 3).
        """
        # Check for correct number of channels based on format
        if format in ("ABGR", "BGRA", "RGBA") and data.shape[-1] != 4:
            raise ValueError(
                "Input data must have four channels for ABGR, BGRA, or RGBA formats"
            )
        elif format in ("RGB", "BGR") and data.shape[-1] != 3:
            raise ValueError(
                "Input data must have three channels for RGB or BGR formats"
            )

        # Handle 4-channel formats
        if format == "ABGR":
            # ABGR -> RGB (select channels 3, 2, 1)
            rgb_data = data[:, :, [3, 2, 1]]
        elif format == "BGRA":
            # BGRA -> RGB (select channels 2, 1, 0)
            rgb_data = data[:, :, [2, 1, 0]]
        elif format == "RGBA":
            # RGBA -> RGB (select channels 0, 1, 2)
            rgb_data = data[:, :, [0, 1, 2]]

        # Handle 3-channel formats
        elif format == "RGB":
            # Already in RGB format, return as is
            rgb_data = data
        elif format == "BGR":
            # BGR -> RGB (select channels 2, 1, 0)
            rgb_data = data[:, :, [2, 1, 0]]
        else:
            raise ValueError(
                "Unsupported format. Expected 'ABGR', 'BGRA', 'RGBA', 'RGB', or 'BGR'."
            )

        return rgb_data

    def _get_rgb_frame(self, info, format: str) -> np.ndarray:
        """
        Extracts the RGB channels from the GStreamer buffer's data in ABGR, BGRA, RGBA, RGB, or BGR format.

        Parameters:
            info (GstVideo.VideoFrame): The GStreamer video frame containing data.
            format (str): The format of the input data. Expected values are 'ABGR', 'BGRA', 'RGBA', 'RGB', or 'BGR'.

        Returns:
            np.ndarray: A new image array with only the RGB channels, shape (height, width, 3).
        """
        if format in ["RGB", "BGR"]:
            # For RGB or BGR formats (3 channels)
            frame = np.ndarray(
                shape=(self.height, self.width, 3),
                dtype=np.uint8,
                buffer=info.data,
            )
            if format == "BGR":
                # Convert BGR to RGB by reordering channels
                frame = frame[:, :, [2, 1, 0]]

        elif format in ["ABGR", "BGRA", "RGBA"]:
            # For formats with 4 channels (ABGR, BGRA, RGBA)
            frame = np.ndarray(
                shape=(self.height, self.width, 4),
                dtype=np.uint8,
                buffer=info.data,
            )
            # Extract RGB using the _extract_rgb method
            frame = self._extract_rgb(frame, format)

        else:
            raise ValueError(
                "Unsupported format. Expected 'ABGR', 'BGRA', 'RGBA', 'RGB', or 'BGR'."
            )

        return frame

    def _get_video_format(self, buffer: Gst.Buffer, pad: Gst.Pad) -> str:
        """
        Retrieves the video format from the GStreamer buffer's caps.

        Parameters:
            buffer (Gst.Buffer): The GStreamer buffer containing video data.
            pad (Gst.Pad): The pad from which to retrieve the video caps.

        Returns:
            str: The video format (e.g., 'RGB', 'RGBA', 'BGRA') or None if not available.
        """
        # Get the caps from the pad
        caps = pad.get_current_caps()
        if not caps:
            caps = pad.get_allowed_caps()

        # Make sure the caps are valid and contain video information
        if not caps or caps.get_size() == 0:
            return None

        # Get the structure of the first caps field (assuming a single format)
        structure = caps.get_structure(0)

        # Check if it's a video format and retrieve the 'format' field
        if structure.has_name("video/x-raw"):
            format_str = structure.get_string("format")
            return format_str

        return None

    def do_transform_ip(self, buf):
        """
        In-place transformation for object detection inference.
        """
        try:
            # Ensure the model is loaded
            if self.get_model() is None:
                self.logger.debug(
                    "do_transform_ip: Model not loaded, calling do_load_model()"
                )
                self.do_load_model()

            # Set a valid timestamp if none is set
            if buf.pts == Gst.CLOCK_TIME_NONE:
                buf.pts = Gst.util_uint64_scale(
                    Gst.util_get_timestamp(),
                    self.framerate_denom,
                    self.framerate_num * Gst.SECOND,
                )

            # Map the buffer to read data
            with buf.map(Gst.MapFlags.READ | Gst.MapFlags.WRITE) as info:
                if info.data is None:
                    self.logger.error(
                        "do_transform_ip: Buffer mapping returned None data."
                    )
                    return Gst.FlowReturn.ERROR

                format = self._get_video_format(buf, self.sinkpad)
                frame = self._get_rgb_frame(info, format)

                # Check if frame is mapped correctly
                if frame is None or not isinstance(frame, np.ndarray):
                    self.logger.error(
                        "do_transform_ip: Frame data is None or not an ndarray."
                    )
                    return Gst.FlowReturn.ERROR
                self.logger.debug(f"do_transform_ip: Frame shape {frame.shape}")

                # Perform inference
                results = self.forward(frame)

                # ✅ Fix: Handle object detection dict separately
                if isinstance(results, dict):
                    self.do_decode(buf, results)  # Directly process the dict
                elif isinstance(results, list):
                    for i, result in enumerate(results):
                        if result is None:
                            self.logger.warning(
                                f"do_transform_ip: Result at index {i} is None, skipping."
                            )
                            continue
                        try:
                            self.do_decode(buf, result)
                        except Exception as e:
                            self.logger.error(
                                f"do_transform_ip: Error in do_decode for result at index {i}: {e}"
                            )
                else:
                    self.logger.error(
                        f"do_transform_ip: Expected dict or list from forward, got {type(results)}."
                    )
                    return Gst.FlowReturn.ERROR

            return Gst.FlowReturn.OK

        except Gst.MapError as e:
            self.logger.error(f"do_transform_ip: Mapping error: {e}")
            return Gst.FlowReturn.ERROR
        except TypeError as e:
            self.logger.error(
                f"do_transform_ip: Type error likely due to NoneType: {e}"
            )
            return Gst.FlowReturn.ERROR
        except Exception as e:
            self.logger.error(
                f"do_transform_ip: Unexpected error during transformation: {e}"
            )
            return Gst.FlowReturn.ERROR

    def do_decode(self, buf, output):
        """
        Decodes the output of the model and adds metadata to the buffer.
        """
        boxes = output["boxes"]
        labels = output["labels"]
        scores = output["scores"]

        # Log buffer pointer and metadata information
        self.logger.info(f"Processing buffer at address: {hex(id(buf))}")
        self.logger.info(f"Processing {len(boxes)} detections.")

        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            x1, y1, x2, y2 = box
            self.logger.info(
                f"Detection {i}: Box coordinates (x1={x1}, y1={y1}, x2={x2}, y2={y2}), "
                f"Label={label}, Score={score:.2f}"
            )

            # Add GstAnalytics metadata to the buffer
            meta = GstAnalytics.buffer_add_analytics_relation_meta(buf)
            if meta:
                qk = GLib.quark_from_string(f"label_{label}")
                ret, mtd = meta.add_od_mtd(qk, x1, y1, x2 - x1, y2 - y1, score)
                if not ret:
                    self.logger.error("Failed to add object detection metadata")
            else:
                self.logger.error("Failed to add GstAnalytics metadata to buffer")

        # Log buffer state after metadata attachment
        attached_meta = GstAnalytics.buffer_get_analytics_relation_meta(buf)
        if not attached_meta:
            self.logger.warning(
                f"Failed to retrieve attached metadata immediately after addition for buffer: {hex(id(buf))}"
            )
