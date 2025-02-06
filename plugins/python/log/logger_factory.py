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
