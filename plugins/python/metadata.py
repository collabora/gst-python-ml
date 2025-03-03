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

import struct
import zlib
from typing import List, Tuple, Any
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

from log.logger_factory import LoggerFactory  # noqa: E402


class Metadata:
    SIGNATURE = b"GST-PYTHON-ML"

    def __init__(self, format_string: str):
        """
        Initialize with a format string:
        - "ifs" -> single struct with int, float, string
        - "l(ifs)" -> list of structs, each with int, float, string
        """
        self.format_string = format_string
        if format_string.startswith("l(") and format_string.endswith(")"):
            self.is_list = True
            self.struct_format = format_string[2:-1]  # Extract "ifs" from "l(ifs)"
        else:
            self.is_list = False
            self.struct_format = format_string

        # Parse the struct format
        self.fixed_fields = [c for c in self.struct_format if c != "s"]
        self.string_count = self.struct_format.count("s")
        self.fixed_size = (
            struct.calcsize("".join(self.fixed_fields)) if self.fixed_fields else 0
        )
        self.meta_bytes = None

    def write(self, buffer: Gst.Buffer, *values) -> None:
        """Add metadata as a Gst.Memory chunk at the end of the buffer."""
        try:
            if self.is_list:
                if not isinstance(values[0], (list, tuple)):
                    raise ValueError(
                        f"Expected a list for format '{self.format_string}', got {type(values[0])}"
                    )
                items = values[0]
                # Pack list length
                meta_bytes = struct.pack("I", len(items))
                # Pack each struct
                for item in items:
                    if len(item) != len(self.struct_format):
                        raise ValueError(
                            f"Struct length mismatch: expected {len(self.struct_format)}, got {len(item)}"
                        )
                    fixed_values = [
                        v for v, f in zip(item, self.struct_format) if f != "s"
                    ]
                    string_values = [
                        v for v, f in zip(item, self.struct_format) if f == "s"
                    ]
                    fixed_bytes = (
                        struct.pack("".join(self.fixed_fields), *fixed_values)
                        if fixed_values
                        else b""
                    )
                    string_bytes = b""
                    for s in string_values:
                        s_bytes = str(s).encode("utf-8")
                        string_bytes += struct.pack("I", len(s_bytes)) + s_bytes
                    meta_bytes += fixed_bytes + string_bytes
            else:
                if len(values) != len(self.struct_format):
                    raise ValueError(
                        f"Value length mismatch: expected {len(self.struct_format)}, got {len(values)}"
                    )
                fixed_values = [
                    v for v, f in zip(values, self.struct_format) if f != "s"
                ]
                string_values = [
                    v for v, f in zip(values, self.struct_format) if f == "s"
                ]
                fixed_bytes = (
                    struct.pack("".join(self.fixed_fields), *fixed_values)
                    if fixed_values
                    else b""
                )
                string_bytes = b""
                for s in string_values:
                    s_bytes = str(s).encode("utf-8")
                    string_bytes += struct.pack("I", len(s_bytes)) + s_bytes
                meta_bytes = fixed_bytes + string_bytes

            # Combine with signature and format string
            self.meta_bytes = (
                self.SIGNATURE
                + struct.pack("I", len(self.format_string))
                + self.format_string.encode("utf-8")
                + meta_bytes
            )

            meta_memory = Gst.Memory.new_wrapped(
                Gst.MemoryFlags.READONLY,
                self.meta_bytes,
                len(self.meta_bytes),
                0,
                len(self.meta_bytes),
                None,
            )
            buffer.append_memory(meta_memory)
            print(f"Metadata added, size: {len(self.meta_bytes)} bytes")
        except struct.error as e:
            raise ValueError(f"Failed to pack values: {str(e)}") from e

    def read(self, buffer: Gst.Buffer) -> tuple:
        """Read metadata from the last memory chunk."""
        n_mem = buffer.n_memory()
        if n_mem < 2:
            raise ValueError(
                f"No metadata memory chunk found (expected at least 2 chunks, got {n_mem})"
            )

        meta_memory = buffer.peek_memory(n_mem - 1)
        with meta_memory.map(Gst.MapFlags.READ) as map_info:
            data_bytes = bytes(map_info.data)
            if not data_bytes.startswith(self.SIGNATURE):
                raise ValueError(
                    f"Invalid metadata signature (expected {self.SIGNATURE.hex()}, got {data_bytes[:len(self.SIGNATURE)].hex()})"
                )

            offset = len(self.SIGNATURE)
            fmt_len = struct.unpack("I", data_bytes[offset : offset + 4])[0]
            offset += 4
            fmt_str = data_bytes[offset : offset + fmt_len].decode("utf-8")
            offset += fmt_len

            if fmt_str != self.format_string:
                raise ValueError(
                    f"Format mismatch: expected {self.format_string}, got {fmt_str}"
                )

            if self.is_list:
                # Read list length
                list_len = struct.unpack("I", data_bytes[offset : offset + 4])[0]
                offset += 4
                result = []
                for _ in range(list_len):
                    fixed_values = []
                    if self.fixed_fields:
                        fixed_bytes = data_bytes[offset : offset + self.fixed_size]
                        fixed_values = list(
                            struct.unpack("".join(self.fixed_fields), fixed_bytes)
                        )
                        offset += self.fixed_size
                    string_values = []
                    for _ in range(self.string_count):
                        str_len = struct.unpack("I", data_bytes[offset : offset + 4])[0]
                        offset += 4
                        string_values.append(
                            data_bytes[offset : offset + str_len].decode("utf-8")
                        )
                        offset += str_len
                    # Combine fixed and string values in struct order
                    struct_values = []
                    fixed_idx = 0
                    string_idx = 0
                    for f in self.struct_format:
                        if f == "s":
                            struct_values.append(string_values[string_idx])
                            string_idx += 1
                        else:
                            struct_values.append(fixed_values[fixed_idx])
                            fixed_idx += 1
                    result.append(tuple(struct_values))
                print(f"Read metadata: {tuple(result)}")
                return tuple(result)
            else:
                fixed_values = []
                if self.fixed_fields:
                    fixed_bytes = data_bytes[offset : offset + self.fixed_size]
                    fixed_values = list(
                        struct.unpack("".join(self.fixed_fields), fixed_bytes)
                    )
                    offset += self.fixed_size
                string_values = []
                for _ in range(self.string_count):
                    str_len = struct.unpack("I", data_bytes[offset : offset + 4])[0]
                    offset += 4
                    string_values.append(
                        data_bytes[offset : offset + str_len].decode("utf-8")
                    )
                    offset += str_len
                # Combine fixed and string values in original order
                result = []
                fixed_idx = 0
                string_idx = 0
                for f in self.struct_format:
                    if f == "s":
                        result.append(string_values[string_idx])
                        string_idx += 1
                    else:
                        result.append(fixed_values[fixed_idx])
                        fixed_idx += 1
                print(f"Read metadata: {tuple(result)}")
                return tuple(result)
