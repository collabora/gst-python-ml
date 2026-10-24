# BaseObjectDetector
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np

PACKED_FORMAT_RGB_SLICES = {
    "RGB": (3, np.s_[:, :, :]),
    "BGR": (3, np.s_[:, :, ::-1]),
    "RGBA": (4, np.s_[:, :, :3]),
    "ARGB": (4, np.s_[:, :, 1:]),
    "BGRA": (4, np.s_[:, :, 2::-1]),
    "ABGR": (4, np.s_[:, :, 3:0:-1]),
}


def packed_format_layout(format: str):
    if format not in PACKED_FORMAT_RGB_SLICES:
        raise ValueError(
            f"Unsupported format {format}, expected one of "
            f"{', '.join(PACKED_FORMAT_RGB_SLICES)}"
        )
    return PACKED_FORMAT_RGB_SLICES[format]


class FormatConverter:
    """Handles video format conversion and extraction for RGB data."""

    @staticmethod
    def extract_rgb(data: np.ndarray, format: str) -> np.ndarray:
        channel_count, rgb_slice = packed_format_layout(format)
        if data.shape[-1] != channel_count:
            raise ValueError(f"{format} data must have {channel_count} channels")
        return data[rgb_slice]

    def get_rgb_frame(self, info, format: str, height: int, width: int) -> np.ndarray:
        channel_count, _ = packed_format_layout(format)
        frame = np.ndarray(
            shape=(height, width, channel_count),
            dtype=np.uint8,
            buffer=info.data,
        )
        return self.extract_rgb(frame, format)

    @staticmethod
    def get_video_format(buffer, pad) -> str:
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
