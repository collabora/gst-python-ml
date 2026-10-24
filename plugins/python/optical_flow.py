# Optical Flow
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from log.global_logger import GlobalLogger
import backend

CAN_REGISTER_ELEMENT = True
try:
    from video_transform import VideoTransform
    from utils.format_converter import FormatConverter
    from engine.optical_flow_engine import ExportedOpticalFlow, OpticalFlowEngine
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
        self.task_engine_name = self.mgr.engine_name = "mp_optical_flow_engine"
        EngineFactory.register(self.mgr.engine_name, OpticalFlowEngine)
        self.format_converter = FormatConverter()
        self._prev_frame = None

    def export_model(self, model_name):
        return ExportedOpticalFlow(model_name, self.mgr.engine_name)

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
        "mp_optical_flow", OpticalFlowTransform
    )
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'mp_optical_flow' element will not be registered because required modules are missing."
    )
