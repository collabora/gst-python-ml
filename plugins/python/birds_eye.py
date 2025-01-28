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
    import os
    import sys

    # Dynamically add the birdseye directory and yolov5 directory to sys.path
    birdseye_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "birdseye"))
    yolov5_path = os.path.join(birdseye_path, "yolov5")
    utils_path = os.path.join(yolov5_path, "utils")

    if birdseye_path not in sys.path:
        sys.path.append(birdseye_path)
    if yolov5_path not in sys.path:
        sys.path.append(yolov5_path)
    if utils_path not in sys.path:
        sys.path.append(utils_path)

    from birds_eye_module import BirdsEyeView
    from arguments import Arguments
    from gst_video_transform import GstVideoTransform
    from gi.repository import Gst, GObject
    import numpy as np
    import cv2
except ImportError as e:
    Gst.warning(f"The 'BirdsEye' element cannot be registered because: {e}")
    CAN_REGISTER_ELEMENT = False


class BirdsEye(GstVideoTransform):
    """
    GStreamer element for transforming the entire video frame using BirdsEyeView.
    """

    __gstmetadata__ = (
        "BirdsEye",
        "Filter/Effect/Video",
        "Applies a bird's-eye perspective transformation to video frames.",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    def __init__(self):
        super().__init__()
        self.processor = None
        self.frame_num = 0
        self._initialize_processor()

    def _initialize_processor(self):
        """
        Initializes the BirdsEyeView processor just like main.py.
        """
        try:
            # Debugging sys.path and Arguments
            Gst.warning(f"sys.path: {sys.path}")
            config = Arguments().parse()  # Uses Arguments to provide defaults
            Gst.warning(f"Arguments parsed: {config}")

            # Initialize BirdsEyeView
            self.processor = BirdsEyeView(config)
            Gst.warning("BirdsEyeView initialized successfully.")
        except Exception as e:
            Gst.error(f"Failed to initialize BirdsEyeView: {e}")
            self.processor = None

    def do_transform_ip(self, buf):
        """
        In-place transformation using the BirdsEyeView class.
        """
        if not self.processor:
            Gst.error("BirdsEyeView processor is not initialized.")
            return Gst.FlowReturn.ERROR

        try:
            with buf.map(Gst.MapFlags.READ | Gst.MapFlags.WRITE) as info:
                if info.data is None:
                    Gst.error("Buffer mapping returned None data.")
                    return Gst.FlowReturn.ERROR

                # Debugging frame dimensions
                Gst.warning(
                    f"Frame dimensions: width={self.width}, height={self.height}"
                )

                # Assume frame dimensions are set somewhere (self.width, self.height)
                frame = np.ndarray(
                    shape=(self.height, self.width, 3),
                    dtype=np.uint8,
                    buffer=info.data,
                )

                # Process the frame using BirdsEyeView
                processed_frame = self.processor.process_frame(
                    frame, self.frame_num, self.width, self.height
                )
                self.frame_num += 1

                # Copy processed frame back to the buffer
                np.copyto(frame, processed_frame)

            return Gst.FlowReturn.OK
        except Exception as e:
            Gst.error(f"Unexpected error during transformation: {e}")
            return Gst.FlowReturn.ERROR


if CAN_REGISTER_ELEMENT:
    GObject.type_register(BirdsEye)
    __gstelementfactory__ = ("pyml_birdseye", Gst.Rank.NONE, BirdsEye)
else:
    Gst.warning("Failed to register the 'BirdsEye' element.")
