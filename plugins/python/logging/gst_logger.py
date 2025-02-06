from logger import Logger

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

Gst.init(None)


class GstLogger(Logger):
    """Logger implementation using GStreamer."""

    def error(self, message, *args):
        Gst.error(message % args if args else message)

    def warn(self, message, *args):
        Gst.warning(message % args if args else message)

    def info(self, message, *args):
        Gst.info(message % args if args else message)

    def debug(self, message, *args):
        Gst.debug(message % args if args else message)
