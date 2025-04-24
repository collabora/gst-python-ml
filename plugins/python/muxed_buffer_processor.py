# muxed_buffer_processor.py
# Copyright (C) 2024-2025 Collabora Ltd.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Library General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.

import numpy as np
import gi
from gi.repository import Gst

from format_converter import FormatConverter
from metadata import Metadata


class MuxedBufferProcessor:
    """
    Utility class to handle GStreamer buffers containing single or multiplexed streams.
    Extracts frames and metadata from single-frame buffers (no metadata) or batch buffers
    (metadata in last chunk). Does not perform any processing or metadata attachment.
    """

    def __init__(self, logger, width, height, framerate_num=30, framerate_denom=1):
        """
        Initialize the MuxedBufferProcessor.

        Args:
            logger: Logger instance for logging messages.
            width (int): Frame width.
            height (int): Frame height.
            framerate_num (int): Framerate numerator (default: 30).
            framerate_denom (int): Framerate denominator (default: 1).
        """
        self.logger = logger
        self.width = width
        self.height = height
        self.framerate_num = framerate_num
        self.framerate_denom = framerate_denom
        self.format_converter = FormatConverter()
        self.metadata = Metadata("si")

    def extract_frames(self, buf, sinkpad):
        """
        Extract frames and metadata from a GStreamer buffer.

        Args:
            buf: GStreamer buffer to process.
            sinkpad: Sink pad of the element (used for format detection).

        Returns:
            Tuple:
                - frames: np.ndarray (single frame or batch of frames).
                - id_str: Metadata ID string (None for single-frame mode).
                - num_sources: Number of sources from metadata (1 for single-frame mode).
                - format: Video format string.

            Returns (None, None, None, None) on error.
        """
        self.logger.info(f"Extracting frames from buffer: {hex(id(buf))}")
        try:
            # Set PTS if not present
            if buf.pts == Gst.CLOCK_TIME_NONE:
                buf.pts = Gst.util_uint64_scale(
                    Gst.util_get_timestamp(),
                    self.framerate_denom,
                    self.framerate_num * Gst.SECOND,
                )

            num_chunks = buf.n_memory()
            format = self.format_converter.get_video_format(buf, sinkpad)
            self.logger.info(f"Chunks: {num_chunks}, format: {format}")

            if num_chunks < 1:
                self.logger.error("Buffer has no memory chunks")
                return None, None, None, None

            # Single frame case: no metadata
            if num_chunks == 1:
                self.logger.info("Single frame mode (no metadata)")
                with buf.peek_memory(0).map(Gst.MapFlags.READ) as info:
                    frame = self.format_converter.get_rgb_frame(
                        info, format, self.height, self.width
                    )
                    if frame is None or not isinstance(frame, np.ndarray):
                        self.logger.error("Invalid frame")
                        return None, None, None, None
                    return frame, None, 1, format

            # Batch case: last chunk is metadata
            else:
                self.logger.info(f"Batch mode with {num_chunks} chunks")
                num_frames = num_chunks - 1
                frames = []
                for i in range(num_frames):
                    with buf.peek_memory(i).map(Gst.MapFlags.READ) as info:
                        frame = self.format_converter.get_rgb_frame(
                            info, format, self.height, self.width
                        )
                        if frame is None or not isinstance(frame, np.ndarray):
                            self.logger.error(f"Invalid frame at index {i}")
                            return None, None, None, None
                        frames.append(frame)

                # Read metadata from the last chunk
                id_str, num_sources = self.metadata.read(buf)
                self.logger.info(f"Metadata: ID={id_str}, num_sources={num_sources}")
                if num_sources != num_frames:
                    self.logger.error(
                        f"Metadata num_sources ({num_sources}) does not match frame count ({num_frames})"
                    )
                    return None, None, None, None

                batch_frames = np.stack(frames, axis=0)
                self.logger.info(f"Extracted batch with shape: {batch_frames.shape}")
                return batch_frames, id_str, num_sources, format

        except Exception as e:
            self.logger.error(f"Frame extraction error: {e}")
            return None, None, None, None
