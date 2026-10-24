# GstLogger
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import sys

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402
from .logger import Logger  # noqa: E402

# a pip pygobject without the gst-python overrides has no Gst.info
if not hasattr(Gst, "info"):
    raise ImportError("Gst has no logging functions, gst-python is not installed")

# the caller of error, warning, info or debug
CALLER_DEPTH = 2


class GstLogger(Logger):
    """Logger implementation using GStreamer with caller file and line number."""

    def _log_with_caller(self, log_func, message, *args):
        """
        Logs a message with the correct caller file and line number.
        """
        frame = sys._getframe(CALLER_DEPTH)
        filename = frame.f_code.co_filename.split("/")[-1]
        lineno = frame.f_lineno

        # Format the message with caller info
        log_message = f"{filename}:{lineno} - {message % args if args else message}"

        # Call the correct GStreamer logging function
        log_func(log_message)

    def error(self, message, *args):
        self._log_with_caller(Gst.error, message, *args)

    def warning(self, message, *args):
        self._log_with_caller(Gst.warning, message, *args)

    def info(self, message, *args):
        self._log_with_caller(Gst.info, message, *args)

    def debug(self, message, *args):
        self._log_with_caller(Gst.debug, message, *args)
