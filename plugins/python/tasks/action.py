# ActionTask
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

"""Backend-agnostic action-recognition task.

`ActionTask` runs classification over a window of frames and turns the result
into an optional output frame (a label overlay) plus a metadata dict, using
only numpy/cv2 and the engine. It never touches the buffer and holds no
temporal state; the backend element shell accumulates the frame window, does
the frame read, the frame write, and the metadata attach through the frameio
facade.

Contract expected from the host element:
  self.engine      - the ML engine (provided by MLEngineMixin)
  self.draw_label  - bool overlay flag (a backend-declared property)
"""

from tasks.frame_format import to_format


class ActionTask:
    """Inference + frame/metadata production, independent of any framework."""

    def forward(self, frame_buffer):
        if self.engine:
            return self.engine.do_forward(frame_buffer)
        return None

    def decode(self, frame, result, fmt):
        """Turn a classification result into ``(output_frame_or_None, blob_bytes)``.

        ``output_frame`` is the label-overlaid frame in pixel format ``fmt``
        (or None when no overlay is drawn). ``blob_bytes`` is the serialized
        action metadata, always returned, to be appended by the shell.
        """
        import json

        import cv2

        label = result.get("label", "")
        score = result.get("score", 0.0)
        output = None

        # Draw label before appending read-only metadata memory
        if self.draw_label and label:
            overlay = frame.copy()
            text = f"{label} ({score:.2f})"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.rectangle(overlay, (8, 8), (16 + tw, 16 + th + 8), (0, 0, 0), -1)
            cv2.putText(
                overlay,
                text,
                (12, 12 + th),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            output = to_format(overlay, fmt)

        # Serialize action metadata for the shell to append.
        return output, json.dumps(result).encode("utf-8")
