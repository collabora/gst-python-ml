# AnomalyTask
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

"""Backend-agnostic anomaly-detection task.

`AnomalyTask` runs inference and turns the result into an optional output frame
(a heatmap overlay) plus a metadata dict, using only numpy/cv2 and the engine.
It never touches the buffer; the backend element shell does the frame read,
the frame write, and the metadata attach through the frameio facade.

Contract expected from the host element:
  self.engine        - the ML engine (provided by MLEngineMixin)
  self.threshold     - anomaly threshold (a backend-declared property)
  self.draw_heatmap  - bool overlay flag (a backend-declared property)
"""

from tasks.frame_format import to_format


class AnomalyTask:
    """Inference + frame/metadata production, independent of any framework."""

    def forward(self, frame):
        if self.task_engine:
            return self.task_engine.do_forward(frame, threshold=self.threshold)
        return None

    def decode(self, frame, result, fmt):
        """Turn an inference result into ``(output_frame_or_None, blob_bytes)``.

        ``output_frame`` is the heatmap-overlaid frame in pixel format ``fmt``
        (or None when no overlay is drawn). ``blob_bytes`` is the serialized
        anomaly metadata, always returned, to be appended by the shell.
        """
        import json

        import cv2
        import numpy as np

        is_anomaly = result.get("is_anomaly", False)
        heatmap = result.get("heatmap")
        output = None

        # Draw heatmap overlay when the frame is flagged.
        if self.draw_heatmap and is_anomaly and heatmap is not None:
            H, W = frame.shape[:2]
            heatmap_resized = cv2.resize(heatmap, (W, H))
            heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

            overlay = cv2.addWeighted(frame, 0.6, heatmap_rgb, 0.4, 0)

            # Draw anomaly score text
            score = result.get("score", 0.0)
            text = f"ANOMALY: {score:.3f}"
            cv2.putText(
                overlay,
                text,
                (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

            output = to_format(overlay, fmt)

        meta = {
            "score": result.get("score", 0.0),
            "is_anomaly": is_anomaly,
        }
        return output, json.dumps(meta).encode("utf-8")
