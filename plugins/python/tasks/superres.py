# SuperResTask
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

"""Backend-agnostic super-resolution task.

`SuperResTask` runs inference and resizes the upscaled output back to the
original frame dimensions, using only numpy/cv2 and the engine. It never
touches the buffer; the backend element shell does the frame read and the
frame write through the frameio facade.

Contract expected from the host element:
  self.engine  - the ML engine (provided by MLEngineMixin)
  self.width   - original buffer width (a backend-declared property)
  self.height  - original buffer height (a backend-declared property)
"""

from tasks.frame_format import to_format


class SuperResTask:
    """Inference + frame production, independent of any framework."""

    def forward(self, frame):
        if self.task_engine:
            return self.task_engine.do_forward(frame)
        return None

    def decode(self, upscaled, fmt):
        """Resize the upscaled frame back to original dimensions.

        Returns ``(output_frame_or_None, None)``: the resized frame in pixel
        format ``fmt`` for the shell to write, and no metadata blob.
        """
        import cv2

        # Resize back to original buffer dimensions for in-place compatibility
        resized = cv2.resize(
            upscaled, (self.width, self.height), interpolation=cv2.INTER_LANCZOS4
        )
        output = to_format(resized, fmt)
        return output, None
