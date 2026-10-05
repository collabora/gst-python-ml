# DepthTask
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

"""Backend-agnostic monocular-depth task.

`DepthTask` runs inference and turns a depth map into an optional output frame
(a colorized depth visualization) plus a serialized metadata blob (the uint8
normalized depth map), using only numpy/cv2 and the engine. It never touches
the buffer; the backend element shell does the frame read, the frame write, and
the metadata attach through the frameio facade.

Contract expected from the host element:
  self.engine     - the ML engine (provided by MLEngineMixin)
  self.logger     - logger
  self.visualize  - bool, replace frame with colorized depth (a property)
  self.colormap   - str colormap name (a property)
"""

# cv2 colormap IDs for depth visualization
COLORMAP_IDS = {
    "inferno": 9,
    "jet": 2,
    "viridis": 16,
    "plasma": 18,
    "magma": 13,
}


class DepthTask:
    """Inference + frame/metadata production, independent of any framework."""

    def forward(self, frames):
        if self.task_engine:
            return self.task_engine.do_forward(frames)
        return None

    def decode(self, frame, depth_map, fmt):
        """Normalize depth, optionally visualize, then build metadata.

        Returns ``(output_frame_or_None, blob_bytes)``. ``output_frame`` is the
        colorized depth visualization in pixel format ``fmt`` (or None when not
        visualizing). ``blob_bytes`` is the uint8 normalized depth map, always
        returned, to be appended by the shell.
        """
        import cv2
        import numpy as np

        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max > d_min:
            depth_norm = ((depth_map - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            depth_norm = np.zeros_like(depth_map, dtype=np.uint8)

        output = None

        # Visualize first, before appending any read-only metadata memory.
        # (A READONLY chunk on the buffer would prevent buf.map(WRITE) from succeeding.)
        if self.visualize:
            cmap_id = COLORMAP_IDS.get(self.colormap, COLORMAP_IDS["inferno"])
            depth_bgr = cv2.applyColorMap(depth_norm, cmap_id)
            output = self._convert_bgr_to_format(depth_bgr, fmt)

        # Build uint8 depth map metadata bytes (payload only; header added by shell).
        depth_bytes = depth_norm.tobytes()
        return output, depth_bytes

    @staticmethod
    def _convert_bgr_to_format(bgr, fmt):
        """Convert a BGR numpy array to the target GStreamer video format."""
        import cv2
        import numpy as np

        if fmt == "RGB":
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif fmt == "BGR":
            return bgr
        elif fmt == "RGBA":
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
        elif fmt == "BGRA":
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        elif fmt == "ARGB":
            rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
            return np.roll(rgba, 1, axis=-1)  # RGBA -> ARGB
        elif fmt == "ABGR":
            bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
            return np.roll(bgra, 1, axis=-1)  # BGRA -> ABGR
        else:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
