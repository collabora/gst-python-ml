# LoggerFactory
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


from .logger import Logger, PythonLogger
import logging

GST_LOGGER_AVAILABLE = False

# Try importing GstLogger, set the flag if successful
try:
    from .gst_logger import GstLogger

    GST_LOGGER_AVAILABLE = True
except ImportError:
    GstLogger = None  # Avoid NameError if accessed
    GST_LOGGER_AVAILABLE = False


class LoggerFactory:
    """Factory for creating logger instances."""

    LOGGER_TYPE_GST = "gst"
    LOGGER_TYPE_PYTHON = "python"

    @staticmethod
    def get(logger_type: str = LOGGER_TYPE_GST) -> Logger:
        """Return an instance of the requested logger.

        If 'gst' is requested but unavailable, warns and falls back to PythonLogger.
        """
        if logger_type == LoggerFactory.LOGGER_TYPE_GST:
            if GST_LOGGER_AVAILABLE:
                return GstLogger()
            else:
                logging.warning(
                    "GStreamer logging is unavailable. Falling back to PythonLogger."
                )
                return PythonLogger()

        elif logger_type == LoggerFactory.LOGGER_TYPE_PYTHON:
            return PythonLogger()

        else:
            raise ValueError(f"Unknown logger type: {logger_type}")
