# OpticalFlowTask
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

"""Backend-agnostic optical-flow task.

`OpticalFlowTask` runs inference on a pair of frames and renders the resulting
flow field as a color overlay, using only numpy/cv2 and the engine. It never
touches the buffer; the backend element shell does the frame read, the frame
write, and the temporal frame-pairing (the previous-frame state) itself.

Contract expected from the host element:
  self.engine  - the ML engine (provided by MLEngineMixin)
"""

from tasks.frame_format import to_format


class OpticalFlowTask:
    """Inference + flow visualization, independent of any framework."""

    def forward(self, prev_frame, curr_frame):
        if self.task_engine:
            return self.task_engine.do_forward(prev_frame, curr_frame)
        return None

    def decode(self, flow, frame, fmt):
        """Render flow as a color overlay blended onto ``frame``.

        Returns ``(output_frame_or_None, None)``: the blended frame in pixel
        format ``fmt`` for the shell to write, and no metadata blob.
        """
        import cv2

        flow_vis = self._flow_to_color(flow)
        blended = cv2.addWeighted(frame, 0.5, flow_vis, 0.5, 0)
        output = to_format(blended, fmt)
        return output, None

    @staticmethod
    def _flow_to_color(flow):
        """Convert optical flow (H, W, 2) to an RGB color image using HSV encoding."""
        import cv2
        import numpy as np

        fx, fy = flow[..., 0], flow[..., 1]
        mag = np.sqrt(fx**2 + fy**2)
        ang = np.arctan2(fy, fx)

        hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
        hsv[..., 0] = ((ang + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
        hsv[..., 1] = 255
        mag_norm = mag / (mag.max() + 1e-8)
        hsv[..., 2] = (mag_norm * 255).astype(np.uint8)

        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
