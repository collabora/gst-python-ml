# BaseClassifier
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


from video_transform import VideoTransform

from backend import analytics


class BaseClassifier(VideoTransform):
    """
    GStreamer element for image classification with a machine learning model.
    """

    def __init__(self):
        super().__init__()
        self.logger.info("BaseClassifier initialized.")

    def do_forward(self, frame):
        """
        Runs classification inference and returns a label with a confidence score.
        """
        if self.task_engine:
            return self.task_engine.do_forward(frame)
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
        if not isinstance(output, dict):
            self.logger.error(
                f"Unexpected classification output format: {type(output)}"
            )
            return

        label = output.get("label")
        score = output.get("score")

        if label is None or score is None:
            self.logger.warning("Classification result missing label or score.")
            return

        self.logger.info(f"Classified as {label} with confidence score {score:.2f}")

        # Attach classification metadata
        meta = analytics.add_relation_meta(buf)
        if meta:
            analytics.add_object(meta, label, 0, 0, self.width, self.height, score)
            self.logger.info(f"Classified as {label} with score {score:.2f}")
