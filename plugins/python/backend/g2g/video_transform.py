# VideoTransform (g2g backend)
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Library General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.

"""g2g backend for video transform elements.

The host calls `g2g_process(buf, w, h, fmt, sink)` once per frame. This base
binds the frame's sink onto the shared `frameio` / `analytics` (so leaf task code
that calls them through `from backend import frameio, analytics` reaches this
frame's buffer and sink), extracts the frame, then hands off to `process_frames`,
the framework-agnostic per-frame hook the leaf element supplies (the same hook the
gst backend drives from `do_transform_ip`).
"""

from backend.g2g.analytics import analytics
from backend.g2g.frameio import frameio
from backend.g2g.transform import BaseTransform


class VideoTransform(BaseTransform):
    """Base for g2g video transform elements."""

    def __init__(self):
        super().__init__()
        self.width = 0
        self.height = 0

    def process_frames(self, frames, num_sources, fmt, target):
        """Run inference + write results for one frame (or batch of `num_sources`).

        `frames` is an (H, W, C) array (single source) or (N, H, W, C); `target`
        is the buffer handle to write a frame back to and to attach metadata to
        (via `frameio` / `analytics`). The concrete element provides this; the
        same hook is shared with the gst backend.
        """
        raise NotImplementedError("leaf element must implement process_frames")

    def g2g_process(self, buf, width, height, fmt, sink):
        self.width = width
        self.height = height
        # Route this frame's pixels (buf) and metadata sink to the shared I/O the
        # leaf task code uses.
        frameio.bind(sink, fmt)
        analytics.bind(sink)
        self._ensure_model()
        frames, num_sources, fmt = frameio.read_frames(buf, None, width, height)
        if frames is None:
            return None
        # The ML elements are RGB-native (in gst this is guaranteed upstream by
        # videoconvert + RGB sink caps). The g2g host carries 4-channel packed
        # formats (RGBA / BGRA); with no convert element in the chain, reduce them
        # to 3-channel RGB here for inference. Detection is read-only, so the
        # original buffer (write-back target) is untouched.
        if frames.ndim == 3 and frames.shape[2] == 4:
            if fmt.upper() in ("BGRA", "ABGR"):
                frames = frames[:, :, 2::-1]  # B,G,R(,A) -> R,G,B
            else:
                frames = frames[:, :, :3]  # R,G,B,A -> R,G,B
        self.process_frames(frames, num_sources, fmt, buf)
        # Blobs/detections are staged on the sink; no return payload needed.
        return None
