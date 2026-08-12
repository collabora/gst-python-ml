# BaseTransform (g2g backend)
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Library General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.

"""g2g backend for the `transform` element family.

The gst counterpart subclasses `GstBase.BaseTransform`; the g2g element is a
plain Python object driven by the host's `g2g_process(...)` call. All engine /
model logic still comes from the portable `MLEngineMixin`, and the shared
tunables come from `ml_property_namespace` declared with the `GObject` shim, so
the same set exists here as on gst.

The model is loaded lazily on the first frame (`_ensure_model`) rather than from
a `do_start` framework virtual, since the g2g host has no start hook.

The payload driver (`g2g_process_payload`, for a stream that is not raw video)
also sits here rather than in the aggregator, so that the aggregator, which
subclasses this, gets it too: a single-chain text or audio element is hosted
1-in-1-out even though its gst counterpart is a `GstBase.Aggregator`.
"""

from backend.core import MLEngineMixin, PayloadProcessingMixin, ml_property_namespace
from backend.g2g.shims import GObject


class BaseTransform(MLEngineMixin, PayloadProcessingMixin):
    """Base for g2g ML transform elements (same in/out format, e.g. detection)."""

    locals().update(ml_property_namespace(GObject))

    def __init__(self):
        self._ml_init()

    def g2g_properties(self):
        """Every property this element declares, for the host to check a pipeline
        line against before it runs.

        Without this the host has no way to tell a knob this element has from a
        typo, and would set the typo as an attribute nothing ever reads.
        """
        return sorted(
            {
                name
                for klass in type(self).__mro__
                for name, value in vars(klass).items()
                if isinstance(value, GObject.Property)
            }
        )

    def _ensure_model(self):
        """Load the model on first use (the g2g host has no start hook).

        Guard on the model, not the engine: setting the `device` property
        eagerly creates the engine (with no model loaded), so an engine-only
        check would skip the load and leave inference with a null model.
        """
        if self.mgr.engine_name and (self.engine is None or self.engine.model is None):
            self.do_load_model()

    def g2g_process_payload(self, buffers, caps, meta):
        """Host driver for a stream that is not raw video: run the element's
        `process_payload` over the input bytes and emit what it returns."""
        self._ensure_model()
        self._ensure_started()
        # copied, not viewed: the host takes its buffer back when this returns,
        # and an element that accumulates keeps what it was given
        for payload in self.process_payload(bytes(memoryview(buffers[0]))):
            meta.emit(payload, duration_ns=self.payload_duration_ns(len(payload)))
