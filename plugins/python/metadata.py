# Metadata
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

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

class Metadata:
    HEADER = b"GST-PYTHON-ML"

    def write(self, buffer: Gst.Buffer, num_sources: int) -> None:
        """Write num_sources with header to the buffer as a memory chunk."""
        metadata_bytes = self.HEADER + num_sources.to_bytes(4, "little")
        metadata_memory = Gst.Memory.new_wrapped(
            Gst.MemoryFlags.READONLY,
            metadata_bytes,
            len(metadata_bytes),
            0,
            len(metadata_bytes),
            None,
        )
        buffer.append_memory(metadata_memory.copy(0, -1))  # Append metadata last

    def read(self, buffer: Gst.Buffer) -> int:
        """Read num_sources from the last memory chunk, verifying the header."""
        if buffer.n_memory() < 1:
            raise ValueError("No memory chunks in buffer")

        last_memory = buffer.peek_memory(buffer.n_memory() - 1)
        with last_memory.map(Gst.MapFlags.READ) as map_info:
            data_bytes = map_info.data
            header_len = len(self.HEADER)
            if len(data_bytes) < header_len + 4:
                raise ValueError(f"Memory chunk too short: {len(data_bytes)} bytes")
            if data_bytes[:header_len] != self.HEADER:
                raise ValueError(f"Invalid metadata header: {data_bytes[:header_len].hex()}")
            return int.from_bytes(data_bytes[header_len:header_len + 4], "little")