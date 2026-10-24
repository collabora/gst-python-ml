# DepthTask
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

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

from functools import cache

# cv2.COLORMAP_* values, kept as numbers so the module imports without cv2
COLORMAP_IDS = {
    "inferno": 14,
    "jet": 2,
    "viridis": 16,
    "plasma": 15,
    "magma": 13,
}


MAXIMUM_DEPTH_LEVEL = 255
DEPTH_LEVEL_COUNT = MAXIMUM_DEPTH_LEVEL + 1


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

        if not isinstance(depth_map, np.ndarray):
            return self.decode_on_the_device(depth_map, fmt)
        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max > d_min:
            depth_norm = (
                (depth_map - d_min) / (d_max - d_min) * MAXIMUM_DEPTH_LEVEL
            ).astype(np.uint8)
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

    def decode_on_the_device(self, depth_map, fmt):
        import torch

        depth_min, depth_max = depth_map.min(), depth_map.max()
        depth_range = depth_max - depth_min
        scaled = (depth_map - depth_min) / depth_range * MAXIMUM_DEPTH_LEVEL
        normalized_depth = torch.where(depth_range > 0, scaled, 0).to(torch.uint8)
        output = None
        if self.visualize:
            colormap_id = COLORMAP_IDS.get(self.colormap, COLORMAP_IDS["inferno"])
            colours = colour_table(colormap_id, fmt, depth_map.device)
            output = colours[normalized_depth.long()]
        return output, normalized_depth.cpu().numpy().tobytes()

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


# applyColorMap colours each depth level independently of its neighbours
@cache
def colour_table(colormap_id, fmt, device):
    import cv2
    import numpy as np
    import torch

    levels = np.arange(DEPTH_LEVEL_COUNT, dtype=np.uint8).reshape(DEPTH_LEVEL_COUNT, 1)
    colours = DepthTask._convert_bgr_to_format(
        cv2.applyColorMap(levels, colormap_id), fmt
    )
    return torch.from_numpy(colours.reshape(DEPTH_LEVEL_COUNT, -1)).to(device)
