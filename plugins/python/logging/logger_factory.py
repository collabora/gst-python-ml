from logger import Logger, PythonLogger
import logging

GST_LOGGER_AVAILABLE = False

# Try importing GstLogger, set the flag if successful
try:
    from gst_logger import GstLogger

    GST_LOGGER_AVAILABLE = True
except ImportError:
    GstLogger = None  # Avoid NameError if accessed
    GST_LOGGER_AVAILABLE = False


class LoggerFactory:
    """Factory for creating logger instances."""

    @staticmethod
    def get_logger(logger_type: str = "gst") -> Logger:
        """Return an instance of the requested logger.

        If 'gst' is requested but unavailable, warns and falls back to PythonLogger.
        """
        if logger_type == "gst":
            if GST_LOGGER_AVAILABLE:
                return GstLogger()
            else:
                logging.warning(
                    "GStreamer logging is unavailable. Falling back to PythonLogger."
                )
                return PythonLogger()

        elif logger_type == "python":
            return PythonLogger()

        else:
            raise ValueError(f"Unknown logger type: {logger_type}")
