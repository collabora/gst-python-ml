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
    GStreamer element for generating a bird's-eye view of video frames.
    """

    __gstmetadata__ = (
        "BirdsEye",
        "Filter/Effect/Video",
        "Applies a bird's-eye perspective transformation to video frames.",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    src_points = GObject.Property(
        type=str,
        default="0,0;640,0;640,480;0,480",
        nick="Source Points",
        blurb="Comma-separated coordinates for the source points (e.g., 'x1,y1;x2,y2;x3,y3;x4,y4')",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        # Default source points: entire frame
        self.src_points_array = np.array(
            [[0, 0], [640, 0], [640, 480], [0, 480]], dtype=np.float32
        )

    def do_set_property(self, prop, value):
        """Set the properties of the object."""
        if prop.name == "src_points":
            Gst.info(f"Setting src_points to: {value}")
            self.src_points = value
            self.src_points_array = self._parse_src_points(value)
        else:
            raise AttributeError(f"Unknown property {prop.name}")

    def do_get_property(self, prop):
        """Get the properties of the object."""
        if prop.name == "src_points":
            return self.src_points
        else:
            raise AttributeError(f"Unknown property {prop.name}")

    def _parse_src_points(self, value):
        """
        Parses the source points string into a NumPy array.

        Parameters:
            value (str): Comma-separated coordinates for the source points.

        Returns:
            np.ndarray: A 4x2 NumPy array representing the source points.
        """
        try:
            points = [
                [int(coord) for coord in pair.split(",")] for pair in value.split(";")
            ]
            if len(points) != 4:
                raise ValueError("Exactly 4 source points are required.")
            return np.array(points, dtype=np.float32)
        except Exception as e:
            Gst.error(f"Invalid source points format: {value}. Error: {e}")
            return self.src_points_array  # Fallback to default points

    def _apply_birds_eye_transform(self, frame):
        """
        Applies the bird's-eye view transformation.

        Parameters:
            frame (np.ndarray): Input frame.

        Returns:
            np.ndarray: Transformed frame.
        """
        # Destination points match the input frame dimensions
        height, width = frame.shape[:2]
        dst_points = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )

        # Compute the perspective transform matrix
        matrix = cv2.getPerspectiveTransform(self.src_points_array, dst_points)

        # Apply the warp perspective to create the bird's-eye view
        transformed_frame = cv2.warpPerspective(frame, matrix, (width, height))
        return transformed_frame

    def do_transform_ip(self, buf):
        """
        In-place transformation for generating the bird's-eye view.
        """
        try:
            if self.src_points_array is None:
                Gst.error("do_transform_ip: Source points are not set.")
                return Gst.FlowReturn.ERROR

            # Set a valid timestamp if none is set
            if buf.pts == Gst.CLOCK_TIME_NONE:
                buf.pts = Gst.util_uint64_scale(
                    Gst.util_get_timestamp(),
                    self.framerate_denom,
                    self.framerate_num * Gst.SECOND,
                )

            # Map the buffer to read/write data
            with buf.map(Gst.MapFlags.READ | Gst.MapFlags.WRITE) as info:
                if info.data is None:
                    Gst.error("do_transform_ip: Buffer mapping returned None data.")
                    return Gst.FlowReturn.ERROR

                # Get the video frame from the buffer
                frame = np.ndarray(
                    shape=(self.height, self.width, 3),
                    dtype=np.uint8,
                    buffer=info.data,
                )

                # Apply the bird's-eye view transformation
                transformed_frame = self._apply_birds_eye_transform(frame)

                # Write the transformed frame back to the buffer
                np.copyto(frame, transformed_frame)

            return Gst.FlowReturn.OK

        except Gst.MapError as e:
            Gst.error(f"do_transform_ip: Mapping error: {e}")
            return Gst.FlowReturn.ERROR
        except Exception as e:
            Gst.error(f"do_transform_ip: Unexpected error during transformation: {e}")
            return Gst.FlowReturn.ERROR


if CAN_REGISTER_ELEMENT:
    GObject.type_register(BirdsEye)
    __gstelementfactory__ = ("pyml_birdseye", Gst.Rank.NONE, BirdsEye)
else:
    Gst.warning("Failed to register the 'pyml_birdseye' element.")
