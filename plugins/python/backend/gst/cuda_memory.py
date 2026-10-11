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
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache
from types import SimpleNamespace

from utils.format_converter import packed_format_layout

GSTREAMER_LIBRARY = "libgstreamer-1.0.so.0"
GST_CUDA_LIBRARY = "libgstcuda-1.0.so.0"
CUDA_DRIVER_LIBRARY = "libcuda.so.1"

GST_MAP_READ = 1
# the gi map override drops this flag, hence ctypes
GST_MAP_CUDA = 1 << 17
MAP_INFO_USER_DATA_SLOTS = 4
GST_PADDING = 4

CUDA_ARRAY_INTERFACE_VERSION = 3
UNSIGNED_BYTE_TYPESTR = "|u1"

CUDA_SUCCESS = 0
CU_EVENT_DISABLE_TIMING = 0x2
STREAM_WAIT_EVENT_FLAGS = 0
CUDA_DRIVER_SIGNATURES = {
    "cuCtxPushCurrent_v2": [ctypes.c_void_p],
    "cuCtxPopCurrent_v2": [ctypes.POINTER(ctypes.c_void_p)],
    "cuEventCreate": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint],
    "cuEventRecord": [ctypes.c_void_p, ctypes.c_void_p],
    "cuStreamWaitEvent": [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint],
    "cuEventDestroy_v2": [ctypes.c_void_p],
    "cuGetErrorName": [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)],
    "cuDeviceGet": [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
    "cuDevicePrimaryCtxRetain": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int],
}


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


class GstMiniObject(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_size_t),
        ("refcount", ctypes.c_int),
        ("lockstate", ctypes.c_int),
        ("flags", ctypes.c_uint),
        ("copy", ctypes.c_void_p),
        ("dispose", ctypes.c_void_p),
        ("free", ctypes.c_void_p),
        ("priv_uint", ctypes.c_uint),
        ("priv_pointer", ctypes.c_void_p),
    ]


class GstMemory(ctypes.Structure):
    _fields_ = [
        ("mini_object", GstMiniObject),
        ("allocator", ctypes.c_void_p),
        ("parent", ctypes.c_void_p),
        ("maxsize", ctypes.c_size_t),
        ("align", ctypes.c_size_t),
        ("offset", ctypes.c_size_t),
        ("size", ctypes.c_size_t),
    ]


# truncated after context, the only field read
class GstCudaMemory(ctypes.Structure):
    _fields_ = [("mem", GstMemory), ("context", ctypes.c_void_p)]


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
    library.gst_cuda_memory_get_stream.argtypes = [ctypes.c_void_p]
    library.gst_cuda_memory_get_stream.restype = ctypes.c_void_p
    library.gst_cuda_stream_get_handle.argtypes = [ctypes.c_void_p]
    library.gst_cuda_stream_get_handle.restype = ctypes.c_void_p
    library.gst_cuda_context_get_handle.argtypes = [ctypes.c_void_p]
    library.gst_cuda_context_get_handle.restype = ctypes.c_void_p
    return library


@cache
def cuda_driver_library():
    library = ctypes.CDLL(CUDA_DRIVER_LIBRARY)
    for function_name, argument_types in CUDA_DRIVER_SIGNATURES.items():
        function = getattr(library, function_name)
        function.argtypes = argument_types
        function.restype = ctypes.c_int
    return library


def call_cuda_driver(function_name, *arguments):
    library = cuda_driver_library()
    result = getattr(library, function_name)(*arguments)
    if result == CUDA_SUCCESS:
        return
    error_name = ctypes.c_char_p()
    library.cuGetErrorName(result, ctypes.byref(error_name))
    raise RuntimeError(f"{function_name} failed with {error_name.value.decode()}")


@contextmanager
def current_cuda_context(context_handle):
    call_cuda_driver("cuCtxPushCurrent_v2", context_handle)
    try:
        yield
    finally:
        call_cuda_driver("cuCtxPopCurrent_v2", ctypes.byref(ctypes.c_void_p()))


# never released, torch keeps the primary context for the life of the process
@cache
def primary_cuda_context(device_index):
    device = ctypes.c_int()
    call_cuda_driver("cuDeviceGet", ctypes.byref(device), device_index)
    context = ctypes.c_void_p()
    call_cuda_driver("cuDevicePrimaryCtxRetain", ctypes.byref(context), device)
    return context.value


def wait_for_the_producer_stream(memory):
    import torch

    gst_cuda = gst_cuda_library()
    producer_stream = gst_cuda.gst_cuda_memory_get_stream(memory)
    stream_handle = (
        gst_cuda.gst_cuda_stream_get_handle(producer_stream)
        if producer_stream
        else None
    )
    if not stream_handle:
        gst_cuda.gst_cuda_memory_sync(memory)
        return
    gst_context = GstCudaMemory.from_address(memory).context
    context_handle = gst_cuda.gst_cuda_context_get_handle(gst_context)
    event = ctypes.c_void_p()
    with current_cuda_context(context_handle):
        call_cuda_driver("cuEventCreate", ctypes.byref(event), CU_EVENT_DISABLE_TIMING)
    try:
        with current_cuda_context(context_handle):
            call_cuda_driver("cuEventRecord", event, stream_handle)
        torch_stream = torch.cuda.current_stream()
        with current_cuda_context(primary_cuda_context(torch_stream.device.index)):
            call_cuda_driver(
                "cuStreamWaitEvent",
                torch_stream.cuda_stream,
                event,
                STREAM_WAIT_EVENT_FLAGS,
            )
    finally:
        call_cuda_driver("cuEventDestroy_v2", event)


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
        wait_for_the_producer_stream(map_info.memory)
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
