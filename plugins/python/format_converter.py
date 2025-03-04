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
from gi.repository import Gst, GstAnalytics, GstVideo, GObject, GLib  # noqa: E402


class FormatConverter:
    """Handles video format conversion and extraction for RGB data."""

    @staticmethod
    def extract_rgb(data: np.ndarray, format: str) -> np.ndarray:
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

    def get_rgb_frame(self, info, format: str, height: int, width: int) -> np.ndarray:
        """
        Extracts the RGB channels from the GStreamer buffer's data in ABGR, BGRA, RGBA, RGB, or BGR format.

        Parameters:
            info (Gst.MapInfo): The mapped buffer memory info.
            format (str): The format of the input data. Expected values are 'ABGR', 'BGRA', 'RGBA', 'RGB', or 'BGR'.
            height (int): The height of the video frame.
            width (int): The width of the video frame.

        Returns:
            np.ndarray: A new image array with only the RGB channels, shape (height, width, 3).
        """
        if format in ["RGB", "BGR"]:
            # For RGB or BGR formats (3 channels)
            frame = np.ndarray(
                shape=(height, width, 3),
                dtype=np.uint8,
                buffer=info.data,
            )
            if format == "BGR":
                # Convert BGR to RGB by reordering channels
                frame = frame[:, :, [2, 1, 0]]

        elif format in ["ABGR", "BGRA", "RGBA"]:
            # For formats with 4 channels (ABGR, BGRA, RGBA)
            frame = np.ndarray(
                shape=(height, width, 4),
                dtype=np.uint8,
                buffer=info.data,
            )
            # Extract RGB using the extract_rgb method
            frame = self.extract_rgb(frame, format)

        else:
            raise ValueError(
                "Unsupported format. Expected 'ABGR', 'BGRA', 'RGBA', 'RGB', or 'BGR'."
            )

        return frame

    @staticmethod
    def get_video_format(buffer: Gst.Buffer, pad: Gst.Pad) -> str:
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
