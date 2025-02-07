# ClassifierBase
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
from gi.repository import Gst, GstAnalytics, GObject, GLib  # noqa: E402


class ClassifierBase(VideoTransform):
    """
    GStreamer element for image classification with a machine learning model.
    """

    def __init__(self):
        super().__init__()
        runtime_check_gstreamer_version()
        self.logger.info("ClassifierBase initialized.")

    def forward(self, frame):
        """
        Runs classification inference and returns a label with a confidence score.
        """
        if self.engine:
            return self.engine.forward(frame)
        self.logger.error("No model loaded in ClassifierBase.")
        return None

    def do_transform_ip(self, buf):
        """
        Processes an image and attaches classification metadata.
        """
        try:
            if self.get_model() is None:
                self.logger.debug("Model not loaded, calling do_load_model()")
                self.do_load_model()

            with buf.map(Gst.MapFlags.READ | Gst.MapFlags.WRITE) as info:
                if info.data is None:
                    self.logger.error("Buffer mapping returned None data.")
                    return Gst.FlowReturn.ERROR

                frame = np.array(info.data, dtype=np.uint8).reshape(
                    self.height, self.width, 3
                )

                # Perform classification
                results = self.forward(frame)
                if not results:
                    self.logger.warning("Classification returned no results.")
                    return Gst.FlowReturn.ERROR

                # Process results
                self.do_decode(buf, results)

            return Gst.FlowReturn.OK

        except Exception as e:
            self.logger.error(f"do_transform_ip: Unexpected error: {e}")
            return Gst.FlowReturn.ERROR

    def do_decode(self, buf, output):
        """
        Decodes classification output and attaches metadata.
        """
        if isinstance(output, dict):
            label = output.get("labels")  # ✅ FIXED: Correct key
            score = output.get("scores")  # ✅ FIXED: Correct key

            # Convert NumPy arrays to standard Python types
            if isinstance(label, np.ndarray) and label.size > 0:
                label = int(label[0])
            if isinstance(score, np.ndarray) and score.size > 0:
                score = float(score[0])

        elif isinstance(output, list) and len(output) == 2:
            label, score = output

        else:
            self.logger.error(
                f"Unexpected classification output format: {type(output)}"
            )
            return

        if label is None or score is None:
            self.logger.warning("Classification result missing label or score.")
            return

        self.logger.info(f"Classified as {label} with confidence score {score:.2f}")

        # Attach classification metadata
        meta = GstAnalytics.buffer_add_analytics_relation_meta(buf)
        if meta:
            qk = GLib.quark_from_string(f"class_{label}")
            meta.add_od_mtd(qk, 0, 0, self.width, self.height, score)
            self.logger.info(f"Classified as {label} with score {score:.2f}")
