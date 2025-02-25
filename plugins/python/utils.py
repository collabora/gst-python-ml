# Utils
# Copyright (C) 2024-2025 Collabora Ltd.
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

import subprocess
import struct
from gi.repository import Gst

from log.logger_factory import LoggerFactory


class StreamMetadata:
    def __init__(self, format_string):
        self.format_string = format_string
        self.struct_size = struct.calcsize(format_string)
        self.logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)

    def create(self, buffer, *values):
        """
        Creates structured data and attaches it directly to the provided GstBuffer.

        Args:
            buffer: GstBuffer to attach the memory to
            *values: Values to pack according to format_string

        Raises:
            ValueError: If values don't match the format string in count or type
        """
        try:
            data_bytes = struct.pack(self.format_string, *values)
            memory = Gst.Memory.new_wrapped(
                Gst.MemoryFlags.READONLY,
                data_bytes,
                len(data_bytes),
                0,
                len(data_bytes),
                None,
            )
            buffer.append_memory(memory)
        except struct.error as e:
            raise ValueError(
                f"Failed to pack values into format {self.format_string}: {str(e)}"
            ) from e

    def read(self, memory):
        with memory.map(Gst.MapFlags.READ) as map_info:
            data_bytes = map_info.data
            if len(data_bytes) < self.struct_size:
                raise ValueError(
                    f"Memory chunk too short: {len(data_bytes)} bytes, expected {self.struct_size}"
                )
            return struct.unpack(self.format_string, data_bytes[: self.struct_size])


def runtime_check_gstreamer_version(min_version="1.24"):
    try:
        result = subprocess.run(
            ["gst-launch-1.0", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            version_line = result.stdout.splitlines()[0]
            current_version = version_line.split(" ")[-1]
            if current_version >= min_version:
                return True
            else:
                raise EnvironmentError(
                    f"GStreamer version {min_version} or higher is required, but version {current_version} is installed."
                )
        else:
            raise EnvironmentError(
                "GStreamer is not installed or not found in the PATH."
            )
    except FileNotFoundError:
        raise EnvironmentError("GStreamer is not installed.")
