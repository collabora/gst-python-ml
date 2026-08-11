# Anomaly Detection
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
    from engine.anomaly_engine import AnomalyEngine
    from engine.engine_factory import EngineFactory
    from backend import frameio, FlowReturn, GObject
    from tasks.anomaly import AnomalyTask

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(f"The 'anomaly' element will not be available. Error {e}")

# Header prefix for anomaly detection buffer metadata
ANOMALY_META_HEADER = b"GST-ANOMALY:"


class AnomalyTransform(VideoTransform, AnomalyTask):
    """
    GStreamer element for anomaly detection in video frames.

    Uses a pretrained feature extractor to compute patch-level anomaly scores
    against a reference distribution of normal frames.

    Set reference-path to a .npy file containing precomputed reference features
    from normal samples. When draw-heatmap=True (default), an anomaly heatmap
    is overlaid on frames that exceed the threshold.

    Anomaly scores are always attached as a GST-ANOMALY: memory chunk (JSON).
    """

    __gstmetadata__ = (
        "Anomaly Detection",
        "Transform",
        "Video anomaly detection using feature extraction and PatchCore scoring",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    threshold = GObject.Property(
        type=float,
        default=0.5,
        minimum=0.0,
        maximum=100.0,
        nick="Anomaly Threshold",
        blurb="Anomaly score threshold above which a frame is flagged",
        flags=GObject.ParamFlags.READWRITE,
    )

    reference_path = GObject.Property(
        type=str,
        default="",
        nick="Reference Path",
        blurb="Path to .npy file with reference feature vectors from normal frames",
        flags=GObject.ParamFlags.READWRITE,
    )

    draw_heatmap = GObject.Property(
        type=bool,
        default=True,
        nick="Draw Heatmap",
        blurb="Overlay anomaly heatmap on frames above threshold",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        self.mgr.engine_name = "pyml_anomaly_engine"
        EngineFactory.register(self.mgr.engine_name, AnomalyEngine)
        self.format_converter = FormatConverter()
        self._reference_loaded = False

    @GObject.Property(type=str)
    def engine_name(self):
        """Machine Learning Engine (read-only for this element)."""
        return self.mgr.engine_name

    @engine_name.setter
    def engine_name(self, value):
        raise ValueError("'engine_name' is read-only for pyml_anomaly")

    def do_transform_ip(self, buf):
        try:
            # Load reference features on first transform if path is set
            if not self._reference_loaded and self.reference_path and self.engine:
                self.engine.load_reference(self.reference_path)
                self._reference_loaded = True

            frames, _num_sources, fmt = frameio.read_frames(
                buf, self.sinkpad, self.width, self.height
            )
            if frames is None:
                return FlowReturn.ERROR

            frame = frames[0] if frames.ndim == 4 else frames
            result = self.forward(frame)
            if result is None:
                return FlowReturn.OK

            # Portable task: produce the optional overlay frame and the metadata.
            output, blob = self.decode(frame, result, fmt)
            if output is not None:
                frameio.write_frame(buf, output)
            if blob is not None:
                frameio.append_blob(buf, ANOMALY_META_HEADER, blob)
            return FlowReturn.OK

        except Exception as e:
            self.logger.error(f"Anomaly detection transform error: {e}")
            return FlowReturn.ERROR


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element(
        "pyml_anomaly", AnomalyTransform
    )
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'pyml_anomaly' element will not be registered because required modules are missing."
    )
