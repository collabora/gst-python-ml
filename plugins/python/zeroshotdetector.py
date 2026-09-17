# ZeroShotDetector
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
from backend import GObject
import backend

CAN_REGISTER_ELEMENT = True
try:
    from base_objectdetector import BaseObjectDetector
    from tasks.zero_shot_detector import ZeroShotDetectorTask

    from engine.zero_shot_detector_engine import ZeroShotDetectorEngine
    from engine.engine_factory import EngineFactory

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'pyml_zeroshotdetector' element will not be available. Error {e}"
    )

DEFAULT_MODEL_NAME = "google/owlv2-base-patch16-ensemble"
DEFAULT_CONFIDENCE = 0.1


class ZeroShotDetector(BaseObjectDetector, ZeroShotDetectorTask):
    __gstmetadata__ = (
        "Zero-Shot Object Detector",
        "Transform",
        "Detects the objects named in the labels property, with no training on those classes",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    confidence = GObject.Property(
        type=float,
        default=DEFAULT_CONFIDENCE,
        minimum=0.0,
        maximum=1.0,
        nick="Confidence Threshold",
        blurb="Minimum detection confidence; a zero-shot score is not "
        "comparable to a trained detector's, so tune it per label set",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        self.mgr.engine_name = "pyml_zero_shot_detector_engine"
        EngineFactory.register(self.mgr.engine_name, ZeroShotDetectorEngine)
        self._model_name = DEFAULT_MODEL_NAME
        self._labels_text = ""
        self._labels_list = []

    @GObject.Property(
        type=str,
        default="",
        nick="Labels",
        blurb="Comma-separated list of things to detect, "
        "e.g. 'person, handbag, red car'",
        flags=GObject.ParamFlags.READWRITE,
    )
    def labels(self):
        return self._labels_text

    @labels.setter
    def labels(self, value):
        self._labels_text = value
        self._labels_list = [
            label.strip() for label in value.split(",") if label.strip()
        ]
        self.logger.info(f"Labels set to: {self._labels_list}")

    @GObject.Property(type=str)
    def engine_name(self):
        """Machine Learning Engine (read-only for this element)."""
        return self.mgr.engine_name

    @engine_name.setter
    def engine_name(self, value):
        raise ValueError("'engine_name' is read-only for pyml_zeroshotdetector")

    def do_forward(self, frames):
        if self.engine:
            self.engine.labels = self._labels_list
            self.engine.confidence = self.confidence
        return super().do_forward(frames)


# The class is backend-agnostic: under g2g the host imports this module and
# instantiates ZeroShotDetector directly, so no GObject registration applies.
# GStreamer factory registration runs only under the gst backend.
if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element(
        "pyml_zeroshotdetector", ZeroShotDetector
    )
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'pyml_zeroshotdetector' element will not be registered because required modules are missing."
    )
