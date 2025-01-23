# BirdsEye
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

CAN_REGISTER_ELEMENT = True
try:
    from gst_video_transform import GstVideoTransform
    from gi.repository import Gst, GObject
    import numpy as np
    import cv2
except ImportError as e:
    Gst.warning(f"The 'BirdsEye' element cannot be registered because: {e}")
    CAN_REGISTER_ELEMENT = False


class BirdsEye(GstVideoTransform):
    """
    GStreamer element for transforming the entire video frame to a bird's-eye view.
    """

    __gstmetadata__ = (
        "BirdsEye",
        "Filter/Effect/Video",
        "Applies a bird's-eye perspective transformation to video frames.",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    tilt = GObject.Property(
        type=float,
        default=30.0,
        nick="Tile Angle",
        blurb="Simulated camera tilt angle in degrees for the bird's-eye view.",
    )

    def do_set_property(self, property, value):
        if property.name == "tilt":
            self.tilt = value
        else:
            super().do_set_property(property, value)

    def do_get_property(self, property):
        if property.name == "tilt":
            return self.tilt
        else:
            return super().do_get_property(property)

    def _compute_source_points(self, width, height):
        """
        Compute the source points based on the tilt angle to simulate a bird's-eye view.

        Parameters:
            width (int): Width of the input frame.
            height (int): Height of the input frame.

        Returns:
            np.ndarray: Array of source points.
        """
        offset = int(width * np.tan(np.radians(self.tilt)) / 2)
        src_points = np.array(
            [
                [offset, 0],  # Top-left
                [width - offset, 0],  # Top-right
                [width, height],  # Bottom-right
                [0, height],  # Bottom-left
            ],
            dtype=np.float32,
        )
        return src_points

    def _apply_birds_eye_transform(self, frame):
        """
        Applies the bird's-eye view transformation.

        Parameters:
            frame (np.ndarray): Input frame.

        Returns:
            np.ndarray: Transformed frame.
        """
        height, width = frame.shape[:2]
        src_points = self._compute_source_points(width, height)
        dst_points = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        return cv2.warpPerspective(frame, matrix, (width, height))

    def do_transform_ip(self, buf):
        """
        In-place transformation for generating the bird's-eye view.
        """
        try:
            with buf.map(Gst.MapFlags.READ | Gst.MapFlags.WRITE) as info:
                if info.data is None:
                    Gst.error("Buffer mapping returned None data.")
                    return Gst.FlowReturn.ERROR
                frame = np.ndarray(
                    shape=(self.height, self.width, 3),
                    dtype=np.uint8,
                    buffer=info.data,
                )
                transformed_frame = self._apply_birds_eye_transform(frame)
                np.copyto(frame, transformed_frame)
            return Gst.FlowReturn.OK
        except Exception as e:
            Gst.error(f"Unexpected error during transformation: {e}")
            return Gst.FlowReturn.ERROR


if CAN_REGISTER_ELEMENT:
    GObject.type_register(BirdsEye)
    __gstelementfactory__ = ("pyml_birdseye", Gst.Rank.NONE, BirdsEye)
else:
    Gst.warning("Failed to register the 'BirdsEye' element.")
