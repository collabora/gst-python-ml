# BaseAggregator (g2g backend)
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Library General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.

"""g2g backend for the `aggregator` element family (N inputs -> 1 output).

The host calls `g2g_process_batch([buf, ...], w, h, fmt, sink)` with one
`FrameBuffer` per input. This base reads each into a frame and hands the stacked
batch to `process_batch`, the framework-agnostic hook the leaf supplies.
"""

import numpy as np

from backend.g2g.analytics import analytics
from backend.g2g.frameio import frameio
from backend.g2g.transform import BaseTransform


class BaseAggregator(BaseTransform):
    """Base for g2g ML aggregator elements (input format may differ from output)."""

    def __init__(self):
        super().__init__()
        self.width = 0
        self.height = 0

    def process_batch(self, frames, num_sources, fmt, target):
        """Run inference over the batched inputs. The concrete element supplies
        this (the same hook the gst aggregator drives)."""
        raise NotImplementedError("leaf element must implement process_batch")

    def g2g_process_batch(self, buffers, width, height, fmt, sink):
        self.width = width
        self.height = height
        frameio.bind(sink, fmt)
        analytics.bind(sink)
        self._ensure_model()
        frames = []
        for buf in buffers:
            frame = frameio.read_frame(buf, None, width, height)
            if frame is not None:
                frames.append(frame)
        if not frames:
            return None
        batch = np.stack(frames, axis=0)
        self.process_batch(batch, len(frames), fmt, buffers[0] if buffers else None)
        return None
