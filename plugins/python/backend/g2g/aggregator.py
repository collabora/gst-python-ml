# BaseAggregator (g2g backend)
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Library General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.

"""g2g backend for the `aggregator` element family (N inputs -> 1 output).

The host calls `g2g_process_batch([buf, ...], w, h, fmt, sink)` with one
`FrameBuffer` per input. This base reads each into a frame and hands them to
`process_frames`, the same per-frame hook a video transform fills in: on gst the
N-source case arrives muxed into one buffer instead, but what the element is
given is the same, so it spells the same.
"""

import numpy as np

from backend.core import FrameProcessingMixin
from backend.g2g.analytics import analytics
from backend.g2g.frameio import as_rgb, frameio
from backend.g2g.transform import BaseTransform


class BaseAggregator(BaseTransform, FrameProcessingMixin):
    """Base for g2g ML aggregator elements (input format may differ from output)."""

    def __init__(self):
        super().__init__()
        self.width = 0
        self.height = 0

    def g2g_process_batch(self, buffers, width, height, fmt, sink):
        self.width = width
        self.height = height
        frameio.bind(sink, fmt)
        analytics.bind(sink)
        self._ensure_model()
        self._ensure_started()
        frames = []
        for buf in buffers:
            frame = frameio.read_frame(buf, None, width, height)
            if frame is not None:
                frames.append(frame)
        if not frames:
            return None
        # (H, W, C) for one source and (N, H, W, C) for several, which is what
        # `read_frames` hands a video transform.
        batch = frames[0] if len(frames) == 1 else np.stack(frames, axis=0)
        self.process_frames(as_rgb(batch, fmt), len(frames), fmt, buffers[0])
        return None
