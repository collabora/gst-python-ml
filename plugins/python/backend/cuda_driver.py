# CUDA driver
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
from functools import cache

CUDA_DRIVER_LIBRARY = "libcuda.so.1"

CUDA_SUCCESS = 0
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
