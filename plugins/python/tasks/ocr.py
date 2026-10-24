# OcrTask
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Backend-agnostic OCR task.

`OcrTask` runs text recognition and turns the result into an optional output
frame (recognized text drawn on the frame) plus a serialized metadata blob
(JSON with recognized text and regions), using only numpy/cv2 and the engine.
It never touches the buffer; the backend element shell does the frame read, the
frame write, and the metadata attach through the frameio facade.

Contract expected from the host element:
  self.engine     - the ML engine (provided by MLEngineMixin)
  self.logger     - logger
  self.draw_text  - bool, draw recognized text on the frame (a property)
"""

from tasks.frame_format import to_format


class OcrTask:
    """Inference + frame/metadata production, independent of any framework."""

    def forward(self, frames):
        if self.engine:
            return self.engine.do_forward(frames)
        return None

    def decode(self, frame, result, fmt):
        """Draw recognized text on frame and build OCR metadata.

        Returns ``(output_frame_or_None, blob_bytes_or_None)``. ``output_frame``
        is the text-overlaid frame in pixel format ``fmt`` (or None when not
        drawing). ``blob_bytes`` is the serialized OCR metadata (or None when
        there are no regions), to be appended by the shell.
        """
        import json

        import cv2

        regions = result.get("regions", [])
        output = None

        # Draw text overlays before appending read-only metadata memory
        if self.draw_text and regions:
            overlay = frame.copy()
            for region in regions:
                x, y, w, h = region["x"], region["y"], region["w"], region["h"]
                text = region["text"]
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
                font_scale = max(0.4, min(w / 300.0, 1.0))
                cv2.putText(
                    overlay,
                    text,
                    (x + 4, y + h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            output = to_format(overlay, fmt)

        # Build OCR results bytes (payload only; header added by shell).
        blob = None
        if regions:
            blob = json.dumps(regions).encode("utf-8")
        return output, blob
