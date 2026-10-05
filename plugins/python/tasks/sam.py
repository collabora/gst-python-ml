# SamTask
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

"""Backend-agnostic segmentation task.

`SamTask` runs Segment Anything inference and turns the result into an optional
output frame (colored mask overlays) plus a serialized metadata blob (JSON with
mask scores and shapes), using only numpy/cv2 and the engine. It never touches
the buffer; the backend element shell does the frame read, the frame write, and
the metadata attach through the frameio facade.

Contract expected from the host element:
  self.engine     - the ML engine (provided by MLEngineMixin)
  self.logger     - logger
  self.visualize  - bool, overlay colored masks on the frame (a property)
  self.max_masks  - int, maximum number of masks (a property)
"""

from tasks.frame_format import to_format

# Mask overlay colors (BGR) cycled across detected objects
MASK_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (128, 0, 255),
    (255, 128, 0),
    (0, 128, 255),
    (128, 255, 0),
]


class SamTask:
    """Inference + frame/metadata production, independent of any framework."""

    def forward(self, frames):
        if self.task_engine:
            return self.task_engine.do_forward(frames, max_masks=self.max_masks)
        return None

    def decode(self, frame, result, fmt):
        """Overlay masks on frame and build mask metadata.

        Returns ``(output_frame_or_None, blob_bytes_or_None)``. ``output_frame``
        is the mask-overlaid frame in pixel format ``fmt`` (or None when not
        visualizing). ``blob_bytes`` is the serialized mask metadata (or None
        when there is none), to be appended by the shell.
        """
        import json

        import cv2
        import numpy as np

        raw_masks = result.get("raw_masks")
        mask_info = result.get("masks", [])
        output = None

        # Draw mask overlays before appending read-only metadata memory
        if self.visualize and raw_masks is not None:
            overlay = frame.copy()
            for j in range(min(raw_masks.shape[0], self.max_masks)):
                best_idx = 0
                if raw_masks.ndim == 4 and raw_masks.shape[1] > 1:
                    best_idx = raw_masks[j].sum(axis=(1, 2)).argmax()
                mask = raw_masks[j, best_idx]
                color = MASK_COLORS[j % len(MASK_COLORS)]
                colored = np.zeros_like(overlay)
                colored[:] = color
                mask_bool = mask > 0.5
                overlay[mask_bool] = cv2.addWeighted(
                    overlay[mask_bool], 0.5, colored[mask_bool], 0.5, 0
                )

            output = to_format(overlay, fmt)

        # Build mask metadata bytes (payload only; header added by shell).
        blob = None
        if mask_info:
            blob = json.dumps(mask_info).encode("utf-8")
        return output, blob
