# BaseTransform (GStreamer backend)
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""GStreamer backend for the `transform` element family (same input/output
format, e.g. object detection).

This file is the GStreamer half of the backend split: the element base
(`GstBase.BaseTransform`) and the framework virtuals (`do_start`). All
engine/model logic lives in the portable `MLEngineMixin`, and the shared
tunables come from `ml_property_namespace`, so both backends declare the same
set from one place.
"""

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
from gi.repository import GObject, GstBase  # noqa: E402

from backend.core import MLEngineMixin, ml_property_namespace  # noqa: E402
from backend.gst.errors import post_model_load_error  # noqa: E402


class BaseTransform(GstBase.BaseTransform, MLEngineMixin):
    """
    Base class for GStreamer transform elements that perform
    inference with a machine learning model. This class manages shared properties
    and handles model loading and device management via MLEngine.
    """

    __gstmetadata__ = (
        "BaseTransform",
        "Transform",
        "Generic machine learning model transform element",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    # unpacked here rather than inherited: pygobject installs a property only
    # when it sits in the class's own dict
    locals().update(ml_property_namespace(GObject))

    def __init__(self):
        super().__init__()
        self._ml_init()

    # GStreamer framework virtual: load the model when the element starts, then
    # run whatever the element itself needs starting (the backend-neutral hook).
    def do_start(self):
        try:
            self.do_load_model()
        except Exception as exception:
            post_model_load_error(self, self._model_name, exception)
            return False
        on_start = getattr(self, "on_start", None)
        if on_start:
            on_start()
        return True
