# Optical Flow
# Copyright (C) 2024-2026 Collabora Ltd.
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

from log.global_logger import GlobalLogger
import backend

CAN_REGISTER_ELEMENT = True
try:
    from video_transform import VideoTransform
    from utils.format_converter import FormatConverter
    from engine.optical_flow_engine import OpticalFlowEngine
    from engine.engine_factory import EngineFactory
    from backend import frameio, GObject
    from tasks.optical_flow import OpticalFlowTask

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'optical_flow' element will not be available. Error {e}"
    )

# Colormap names for flow visualization
FLOW_COLORMAPS = {
    "hsv": None,  # custom HSV-based flow coloring
    "jet": 2,
    "viridis": 16,
    "inferno": 9,
}


class OpticalFlowTransform(VideoTransform, OpticalFlowTask):
    """
    GStreamer element for dense optical flow estimation using RAFT.

    Set model-name to a RAFT variant: raft_large or raft_small.

    Computes dense optical flow between consecutive frames. When
    visualize=True (default), the flow is rendered as a color-coded overlay
    on the video frame using HSV color space (hue=direction, value=magnitude).
    """

    __gstmetadata__ = (
        "Optical Flow",
        "Transform",
        "Dense optical flow estimation using RAFT",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    visualize = GObject.Property(
        type=bool,
        default=True,
        nick="Visualize Flow",
        blurb="Overlay color-coded optical flow on the video frame",
        flags=GObject.ParamFlags.READWRITE,
    )

    colormap = GObject.Property(
        type=str,
        default="hsv",
        nick="Colormap",
        blurb="Colormap for flow visualization: hsv, jet, viridis, inferno",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        self.mgr.engine_name = "pyml_optical_flow_engine"
        EngineFactory.register(self.mgr.engine_name, OpticalFlowEngine)
        self.format_converter = FormatConverter()
        self._prev_frame = None

    @GObject.Property(type=str)
    def engine_name(self):
        """Machine Learning Engine (read-only for this element)."""
        return self.mgr.engine_name

    @engine_name.setter
    def engine_name(self, value):
        raise ValueError("'engine_name' is read-only for pyml_optical_flow")

    def process_frames(self, frames, num_sources, fmt, target):
        """Pair this frame with the previous one and draw the flow overlay."""
        frame = frames[0] if frames.ndim == 4 else frames

        # Temporal pairing stays in the shell: hold the previous frame.
        if self._prev_frame is None:
            self._prev_frame = frame.copy()
            return

        flow = self.forward(self._prev_frame, frame)
        self._prev_frame = frame.copy()

        if flow is None:
            return

        if self.visualize:
            # Portable task: render the flow overlay frame.
            output, blob = self.decode(flow, frame, fmt)
            frameio.write_result(target, output, blob)


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element(
        "pyml_optical_flow", OpticalFlowTransform
    )
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'pyml_optical_flow' element will not be registered because required modules are missing."
    )
