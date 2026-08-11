# g2g backend
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Library General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.

"""g2g backend: element bases plus the frame I/O and analytics implementations.

The counterpart of `backend.gst`, targeting the glass2glass (`g2g`) host instead
of GStreamer. Elements are plain Python objects the host drives via
`g2g_process` / `g2g_process_batch`; there is no GObject type system or GstBase
element, so `GObject` / `FlowReturn` are lightweight shims (see `shims.py`) that
let leaf element code load unchanged. Selected by `GSTML_BACKEND=g2g`.

Unlike the gst backend, no `gi` is imported anywhere here, so a leaf element can
be hosted with no GStreamer present at all.
"""

from backend.g2g.shims import GObject, FlowReturn  # noqa: F401
from backend.g2g.analytics import analytics  # noqa: F401
from backend.g2g.frameio import frameio  # noqa: F401
from backend.g2g.transform import BaseTransform  # noqa: F401
from backend.g2g.video_transform import VideoTransform  # noqa: F401
from backend.g2g.aggregator import BaseAggregator  # noqa: F401

__all__ = [
    "BaseTransform",
    "BaseAggregator",
    "VideoTransform",
    "analytics",
    "frameio",
    "FlowReturn",
    "GObject",
]
