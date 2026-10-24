# Anomaly Detection
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
    from engine.anomaly_engine import AnomalyEngine, ExportedAnomaly
    from engine.engine_factory import EngineFactory
    from backend import frameio, GObject
    from tasks.anomaly import AnomalyTask

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(f"The 'anomaly' element will not be available. Error {e}")

# Header prefix for anomaly detection buffer metadata
ANOMALY_META_HEADER = b"GST-ANOMALY:"


DEFAULT_BACKBONE = "wide_resnet50_2"


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

    META_HEADER = ANOMALY_META_HEADER

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
        self.task_engine_name = self.mgr.engine_name = "mp_anomaly_engine"
        EngineFactory.register(self.mgr.engine_name, AnomalyEngine)
        self.model_name = DEFAULT_BACKBONE
        self.format_converter = FormatConverter()
        self._reference_loaded = False

    def export_model(self, model_name):
        return ExportedAnomaly(model_name, self.mgr.engine_name)

    def process_frames(self, frames, num_sources, fmt, target):
        """Score the primary frame against the reference features."""
        if not self._reference_loaded and self.reference_path and self.task_engine:
            self.task_engine.load_reference(self.reference_path)
            self._reference_loaded = True

        frame = frames[0] if frames.ndim == 4 else frames
        result = self.forward(frame)
        if result is None:
            return

        output, blob = self.decode(frame, result, fmt)
        frameio.write_result(target, output, blob, self.META_HEADER)


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element("mp_anomaly", AnomalyTransform)
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'mp_anomaly' element will not be registered because required modules are missing."
    )
