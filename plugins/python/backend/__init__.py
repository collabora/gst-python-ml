# Pluggable element backend
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Library General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.

"""Pluggable multimedia-framework backend for the ML elements.

The ML logic (engines, tasks, base classes) is framework-agnostic. The pieces
that DO depend on the framework are isolated behind this package:

  * the element base classes `BaseTransform` / `BaseAggregator` / `VideoTransform`
  * the `analytics` metadata interface (add/get/remove detections)

Select the active backend with the `PYML_BACKEND` environment variable
(default ``"gst"``). To add another backend, create a sibling package exposing
the same names (reusing `backend.core.MLEngineMixin` and implementing
`backend.analytics.AnalyticsBackend`) and add a branch below.
"""

import os

from backend.analytics import AnalyticsBackend  # noqa: F401  (re-exported)
from backend.frameio import FrameIO  # noqa: F401  (re-exported)
from backend.core import MLEngineMixin  # noqa: F401  (re-exported)

BACKEND = os.environ.get("PYML_BACKEND", "gst").lower()

if BACKEND == "gst":
    from backend.gst import (
        BaseTransform,
        BaseAggregator,
        VideoTransform,
        analytics,
        frameio,
        FlowReturn,
        GObject,
    )
elif BACKEND == "g2g":
    from backend.g2g import (
        BaseTransform,
        BaseAggregator,
        VideoTransform,
        analytics,
        frameio,
        FlowReturn,
        GObject,
    )
else:
    raise ImportError(
        f"Unknown PYML_BACKEND={BACKEND!r}; supported backends: 'gst', 'g2g'"
    )


def register_gst_element(name, cls, type_name=None):
    """Register `cls` with the GObject type system and return its factory tuple.

    Leaf elements call this behind a `BACKEND == "gst"` guard: under g2g the
    element base classes are plain Python objects, so type_register would fail.
    """
    from backend.gst import Gst, GObject

    GObject.type_register(cls, type_name)
    return (name, Gst.Rank.NONE, cls)


__all__ = [
    "BACKEND",
    "register_gst_element",
    "BaseTransform",
    "BaseAggregator",
    "VideoTransform",
    "analytics",
    "frameio",
    "FlowReturn",
    "GObject",
    "AnalyticsBackend",
    "FrameIO",
    "MLEngineMixin",
]
