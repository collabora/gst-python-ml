# BaseClassifier
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


from utils.runtime_utils import runtime_check_gstreamer_version
from video_transform import VideoTransform

from backend import analytics


class BaseClassifier(VideoTransform):
    """
    GStreamer element for image classification with a machine learning model.
    """

    def __init__(self):
        super().__init__()
        runtime_check_gstreamer_version()
        self.logger.info("BaseClassifier initialized.")

    def do_forward(self, frame):
        """
        Runs classification inference and returns a label with a confidence score.
        """
        if self.engine:
            return self.engine.do_forward(frame)
        self.logger.error("No model loaded in BaseClassifier.")
        return None

    def process_frames(self, frames, num_sources, fmt, target):
        """Classify the frame and attach the label as metadata."""
        results = self.do_forward(frames)
        if not results:
            raise RuntimeError("classification returned no results")
        self.do_decode(target, results)

    def do_decode(self, buf, output):
        """
        Decodes classification output and attaches metadata.
        """
        import numpy as np

        if isinstance(output, dict):
            label = output.get("labels")  # e.g., [405]
            score = output.get("scores")  # e.g., [0.057...]

            # Convert to scalars, handling both NumPy arrays and lists
            if isinstance(label, (np.ndarray, list)) and len(label) > 0:
                label = int(label[0])
            if isinstance(score, (np.ndarray, list)) and len(score) > 0:
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
        meta = analytics.add_relation_meta(buf)
        if meta:
            analytics.add_object(
                meta, f"class_{label}", 0, 0, self.width, self.height, score
            )
            self.logger.info(f"Classified as {label} with score {score:.2f}")
