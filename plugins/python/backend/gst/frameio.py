# GstFrameIO
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

"""GStreamer implementation of the frame buffer I/O interface.

Reads use `MuxedBufferProcessor` (handles single and batched/muxed buffers);
writes map the buffer for writing and memmove the frame bytes in place; blob
append wraps the bytes in a new memory chunk on the buffer.
"""

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

from log.logger_factory import LoggerFactory  # noqa: E402
from utils.muxed_buffer_processor import MuxedBufferProcessor  # noqa: E402
from utils.format_converter import FormatConverter  # noqa: E402
from backend.frameio import FrameIO  # noqa: E402


class GstFrameIO(FrameIO):
    def __init__(self):
        self.logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)
        self.format_converter = FormatConverter()

    def read_frames(self, target, source, width, height, framerate=(30, 1)):
        processor = MuxedBufferProcessor(
            self.logger, width, height, framerate[0], framerate[1]
        )
        return processor.extract_frames(target, source)

    def read_frame(self, target, source, width, height):
        success, map_info = target.map(Gst.MapFlags.READ)
        if not success:
            return None
        try:
            return self.format_converter.to_rgb(
                map_info.data, width, height, target, source
            )
        finally:
            target.unmap(map_info)

    def write_frame(self, target, frame):
        import ctypes
        import numpy as np

        success, map_info = target.map(Gst.MapFlags.WRITE)
        if not success:
            return False
        try:
            frame_bytes = np.ascontiguousarray(frame).tobytes()
            dst = (ctypes.c_char * map_info.size).from_buffer(map_info.data)
            ctypes.memmove(dst, frame_bytes, min(len(frame_bytes), map_info.size))
            return True
        finally:
            target.unmap(map_info)

    def append_blob(self, target, header, payload):
        blob = bytes(header) + bytes(payload)
        tmp = Gst.Buffer.new_allocate(None, len(blob), None)
        tmp.fill(0, blob)
        target.append_memory(tmp.get_memory(0))
        return True
