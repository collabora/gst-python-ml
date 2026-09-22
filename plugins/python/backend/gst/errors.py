# Element error messages (GStreamer backend)
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
