# GstLogger
# Copyright (C) 2024-2025 Collabora Ltd.
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

import inspect
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402
from .logger import Logger  # noqa: E402


class GstLogger(Logger):
    """Logger implementation using GStreamer with caller file and line number."""

    def _log_with_caller(self, log_func, message, *args):
        """
        Logs a message with the correct caller file and line number.
        """
        frame = inspect.stack()[2]  # Get the caller (not this method)
        filename = frame.filename.split("/")[-1]  # Get only filename
        lineno = frame.lineno  # Get line number

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
