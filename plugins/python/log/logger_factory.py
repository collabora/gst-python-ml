# LoggerFactory
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


from .logger import Logger, PythonLogger
from utils.selected_backend import BACKEND
import logging


class LoggerFactory:
    """Factory for creating logger instances."""

    LOGGER_TYPE_GST = "gst"
    LOGGER_TYPE_PYTHON = "python"

    @staticmethod
    def get(logger_type: str = LOGGER_TYPE_GST) -> Logger:
        """Return an instance of the requested logger.

        If 'gst' is requested but unavailable, warns and falls back to PythonLogger.
        """
        if logger_type == LoggerFactory.LOGGER_TYPE_GST and BACKEND != "gst":
            return PythonLogger()
        if logger_type == LoggerFactory.LOGGER_TYPE_GST:
            try:
                from .gst_logger import GstLogger
            # gi without gstreamer raises ValueError from require_version
            except (ImportError, ValueError):
                logging.warning(
                    "GStreamer logging is unavailable. Falling back to PythonLogger."
                )
                return PythonLogger()
            return GstLogger()

        elif logger_type == LoggerFactory.LOGGER_TYPE_PYTHON:
            return PythonLogger()

        else:
            raise ValueError(f"Unknown logger type: {logger_type}")
