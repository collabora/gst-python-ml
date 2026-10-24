# Element error messages (GStreamer backend)
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import traceback

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GLib", "2.0")
from gi.repository import GLib, Gst  # noqa: E402


def post_error(element, summary, exception):
    detail = f"{summary}: {type(exception).__name__}: {exception}"
    error = GLib.Error.new_literal(
        Gst.StreamError.quark(), detail, Gst.StreamError.FAILED
    )
    debug = "".join(
        traceback.format_exception(type(exception), exception, exception.__traceback__)
    )
    # nothing to post on before the element joins a bin, so the caller sees the raise
    if not element.post_message(Gst.Message.new_error(element, error, debug)):
        raise exception


def post_model_load_error(element, model_name, exception):
    post_error(element, f"failed to load model {model_name}", exception)
