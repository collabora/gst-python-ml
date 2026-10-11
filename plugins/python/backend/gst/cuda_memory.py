# CUDA memory
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

import ctypes
from dataclasses import dataclass
from functools import cache
from types import SimpleNamespace

from utils.format_converter import packed_format_layout

GSTREAMER_LIBRARY = "libgstreamer-1.0.so.0"
GST_CUDA_LIBRARY = "libgstcuda-1.0.so.0"

GST_MAP_READ = 1
# the gi map override drops this flag, hence ctypes
GST_MAP_CUDA = 1 << 17
MAP_INFO_USER_DATA_SLOTS = 4
GST_PADDING = 4

CUDA_ARRAY_INTERFACE_VERSION = 3
UNSIGNED_BYTE_TYPESTR = "|u1"


class GstMapInfo(ctypes.Structure):
    _fields_ = [
        ("memory", ctypes.c_void_p),
        ("flags", ctypes.c_int),
        ("data", ctypes.c_void_p),
        ("size", ctypes.c_size_t),
        ("maxsize", ctypes.c_size_t),
        ("user_data", ctypes.c_void_p * MAP_INFO_USER_DATA_SLOTS),
        ("padding", ctypes.c_void_p * GST_PADDING),
    ]


@cache
def gstreamer_library():
    library = ctypes.CDLL(GSTREAMER_LIBRARY)
    library.gst_buffer_map.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(GstMapInfo),
        ctypes.c_int,
    ]
    library.gst_buffer_map.restype = ctypes.c_int
    library.gst_buffer_unmap.argtypes = [ctypes.c_void_p, ctypes.POINTER(GstMapInfo)]
    library.gst_buffer_unmap.restype = None
    return library


@cache
def gst_cuda_library():
    library = ctypes.CDLL(GST_CUDA_LIBRARY)
    library.gst_cuda_memory_sync.argtypes = [ctypes.c_void_p]
    library.gst_cuda_memory_sync.restype = None
    return library


@cache
def gst_cuda_module():
    import gi

    try:
        gi.require_version("GstCuda", "1.0")
    except ValueError:
        return None
    from gi.repository import GstCuda

    return GstCuda


def is_cuda_buffer(buffer):
    gst_cuda = gst_cuda_module()
    # without the typelib the gi map downloads cuda memory to the host
    if gst_cuda is None:
        return False
    memories = [buffer.peek_memory(index) for index in range(buffer.n_memory())]
    return bool(memories) and all(
        gst_cuda.is_cuda_memory(memory) for memory in memories
    )


@dataclass
class DeviceFrameMap:
    buffer: object
    map_info: GstMapInfo
    tensor: object

    def unmap(self):
        gstreamer_library().gst_buffer_unmap(
            self.buffer.__ptr__(), ctypes.byref(self.map_info)
        )


def map_device_frame(buffer, height, width, channels):
    import torch

    map_info = GstMapInfo()
    mapped = gstreamer_library().gst_buffer_map(
        buffer.__ptr__(), ctypes.byref(map_info), GST_MAP_READ | GST_MAP_CUDA
    )
    if not mapped:
        raise RuntimeError("could not map the cuda memory of the buffer")
    device_map = DeviceFrameMap(buffer, map_info, None)
    try:
        # the producer may still be writing the frame on its own cuda stream
        gst_cuda_library().gst_cuda_memory_sync(map_info.memory)
        array_interface = SimpleNamespace(
            __cuda_array_interface__={
                "shape": (height, width, channels),
                "typestr": UNSIGNED_BYTE_TYPESTR,
                "data": (map_info.data, False),
                "version": CUDA_ARRAY_INTERFACE_VERSION,
                "strides": None,
            }
        )
        device_map.tensor = torch.as_tensor(array_interface, device="cuda")
    except BaseException:
        device_map.unmap()
        raise
    return device_map


def rgb_device_frame(frame, video_format):
    channel_count, rgb_slice = packed_format_layout(video_format)
    rgb_channels = range(channel_count)[rgb_slice[-1]]
    if rgb_channels.step == 1:
        return frame[..., rgb_channels.start : rgb_channels.stop]
    # torch refuses negative strides
    return frame[..., list(rgb_channels)]
