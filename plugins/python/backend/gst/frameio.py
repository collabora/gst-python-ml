# GstFrameIO
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from contextlib import contextmanager

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

from log.logger_factory import LoggerFactory  # noqa: E402
from utils.blobs import BLOB_PREFIX  # noqa: E402
from utils.metadata import first_memory_with_prefix  # noqa: E402
from utils.muxed_buffer_processor import MuxedBufferProcessor  # noqa: E402
from utils.format_converter import FormatConverter, packed_format_layout  # noqa: E402
from backend.frameio import FrameIO  # noqa: E402
from backend.gst.cuda_memory import (  # noqa: E402
    MAPPED_MEMORY_COUNT,
    VIDEO_MEMORY,
    DeviceFrameMap,
    is_cuda_buffer,
    map_device_frame,
    rgb_device_frame,
    write_device_frame,
)


# mapping the whole buffer merges the appended metadata into the frame
def map_video_memory(buffer, flags):
    return buffer.map_range(VIDEO_MEMORY, MAPPED_MEMORY_COUNT, flags)


@contextmanager
def mapped_frame(buffer, flags, width, height, bytes_per_pixel):
    import numpy as np

    success, map_info = map_video_memory(buffer, flags)
    if not success:
        yield None
        return
    try:
        count = width * height * bytes_per_pixel
        frame = np.frombuffer(map_info.data, dtype=np.uint8, count=count)
        yield frame.reshape(height, width, bytes_per_pixel)
    finally:
        buffer.unmap(map_info)


class GstFrameIO(FrameIO):
    def __init__(self):
        self.logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)
        self.format_converter = FormatConverter()
        # the frames read from a buffer are views into these maps
        self.read_map_infos = {}

    def keep_mapped(self, target, map_infos):
        self.read_map_infos.setdefault(target, []).extend(map_infos)

    def read_frames(self, target, source, width, height, framerate=(30, 1)):
        if is_cuda_buffer(target):
            return self.read_device_frame(target, source, width, height)
        processor = MuxedBufferProcessor(
            self.logger, width, height, framerate[0], framerate[1]
        )
        try:
            return processor.extract_frames(target, source)
        finally:
            self.keep_mapped(target, processor.map_infos)

    def read_device_frame(self, target, source, width, height):
        first_blob = first_memory_with_prefix(target, BLOB_PREFIX)
        frame_memories = target.n_memory() if first_blob is None else first_blob
        if frame_memories != MAPPED_MEMORY_COUNT:
            raise ValueError(
                f"a buffer of {frame_memories} cuda memories is a batch, "
                "frames in cuda memory are read one per buffer"
            )
        video_format = self.format_converter.get_video_format(target, source)
        channel_count, _ = packed_format_layout(video_format)
        device_map = map_device_frame(target, height, width, channel_count)
        self.keep_mapped(target, [device_map])
        return rgb_device_frame(device_map.tensor, video_format), 1, video_format

    # all channels, torch copies a reversed channel view and drops the writes
    def read_frame_for_drawing(self, target, width, height, video_format):
        import numpy as np

        channel_count, _ = packed_format_layout(video_format)
        if is_cuda_buffer(target):
            device_map = map_device_frame(
                target, height, width, channel_count, writable=True
            )
            self.keep_mapped(target, [device_map])
            return device_map.tensor
        # gstreamer 1.24's binding has no is_writable, a write map fails instead
        success, map_info = map_video_memory(target, Gst.MapFlags.WRITE)
        if not success:
            raise ValueError(
                "drawing in place needs a writable buffer, this one is shared"
            )
        self.keep_mapped(target, [map_info])
        frame = np.frombuffer(
            map_info.data, dtype=np.uint8, count=width * height * channel_count
        )
        return frame.reshape(height, width, channel_count)

    def read_frame(self, target, source, width, height):
        success, map_info = map_video_memory(target, Gst.MapFlags.READ)
        if not success:
            return None
        self.keep_mapped(target, [map_info])
        video_format = self.format_converter.get_video_format(target, source)
        return self.format_converter.get_rgb_frame(
            map_info, video_format, height, width
        )

    def release_frames(self, target):
        for map_info in self.read_map_infos.pop(target, []):
            if isinstance(map_info, DeviceFrameMap):
                map_info.unmap()
            else:
                target.unmap(map_info)

    def write_frame(self, target, frame):
        import numpy as np

        if is_cuda_buffer(target):
            self.release_frames(target)
            write_device_frame(target, frame)
            return True
        read_memories = [
            np.frombuffer(map_info.data, dtype=np.uint8)
            for map_info in self.read_map_infos.get(target, [])
        ]
        if any(np.shares_memory(frame, memory) for memory in read_memories):
            return True
        # a write map over an open read map makes gstreamer copy the memory
        self.release_frames(target)
        success, map_info = map_video_memory(target, Gst.MapFlags.WRITE)
        if not success:
            return False
        try:
            memory = np.frombuffer(map_info.data, dtype=np.uint8)
            if frame.size > memory.size:
                raise ValueError(
                    f"a frame of {frame.size} values does not fit in "
                    f"{memory.size} bytes of video memory"
                )
            np.copyto(memory[: frame.size].reshape(frame.shape), frame)
            return True
        finally:
            target.unmap(map_info)

    def append_blob(self, target, header, payload):
        blob = bytes(header) + bytes(payload)
        tmp = Gst.Buffer.new_allocate(None, len(blob), None)
        tmp.fill(0, blob)
        target.append_memory(tmp.get_memory(0))
        return True
