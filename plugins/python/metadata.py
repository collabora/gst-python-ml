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
    """
    A class for embedding and retrieving structured metadata into/from GStreamer buffers.

    This class supports packing a list of values into a binary format, optionally compressing it,
    and attaching it as memory to a GStreamer buffer. It also allows reading the metadata back
    from a buffer, validating its signature, version, and structure.

    Example usage:
        ```python
        # Initialize GStreamer
        Gst.init(None)

        # Create a Metadata instance
        metadata = Metadata(format_string="IfLs", name="example", include_format=True, compress=False)

        # Create a buffer and add metadata
        buffer = Gst.Buffer.new()
        values = [(1, 2.5, "test"), (2, 3.7, "example")]
        metadata.create(buffer, values)

        # Read metadata back from the buffer
        read_values = metadata.read(buffer)
        print(read_values)  # Output: [(1, 2.5, 'test'), (2, 3.7, 'example')]
        ```

    Attributes:
        SIGNATURE (bytes): A fixed signature identifying the metadata ("GST-PYTHON-ML").
        VERSION (int): The version of the metadata format (currently 1).
        FORMAT_LENGTH_SIZE (int): Size in bytes of the format string length field (4).
        NAME_LENGTH_SIZE (int): Size in bytes of the name length field (4).
        COUNT_SIZE (int): Size in bytes of the struct count field (4).
        ALIGNMENT (int): Byte alignment for padding (4).
    """

    SIGNATURE = b"GST-PYTHON-ML"
    VERSION = 1
    FORMAT_LENGTH_SIZE = 4
    NAME_LENGTH_SIZE = 4
    COUNT_SIZE = 4
    ALIGNMENT = 4

    def __init__(
        self,
        format_string: str,
        include_format: bool = True,
        compress: bool = False,
        name: str = "",
    ):
        """
        Initialize the Metadata instance.

        Args:
            format_string (str): A struct format string (e.g., "IfLs" for int, float, string).
            include_format (bool, optional): Whether to include the format string in the metadata.
                Defaults to True.
            compress (bool, optional): Whether to compress the metadata using zlib. Defaults to False.
            name (str, optional): A name identifier for the metadata. Defaults to "".
        """
        self.format_string = format_string
        self.struct_size = self._calc_struct_size(format_string)
        self.logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)
        self.include_format = include_format
        self.compress = compress
        self.name = name
        self.name_bytes = name.encode("utf-8") if name else b""
        self.name_length = len(self.name_bytes)

        if include_format:
            self.format_bytes = format_string.encode("utf-8")
            self.format_length = len(self.format_bytes)
            self.core_size = (
                self.NAME_LENGTH_SIZE
                + self.name_length
                + self.FORMAT_LENGTH_SIZE
                + self.format_length
                + self.COUNT_SIZE
            )
        else:
            self.core_size = self.NAME_LENGTH_SIZE + self.name_length + self.COUNT_SIZE

        self.signature_size = self._aligned_size(len(self.SIGNATURE) + 2)

    def _aligned_size(self, size: int) -> int:
        """Calculate the size aligned to the nearest multiple of ALIGNMENT."""
        return size + ((self.ALIGNMENT - (size % self.ALIGNMENT)) % self.ALIGNMENT)

    def _pad_to(self, data: bytes, alignment: int = ALIGNMENT) -> bytes:
        """Pad data with null bytes to the specified alignment."""
        padding = (alignment - (len(data) % alignment)) % alignment
        return data + b"\x00" * padding

    def _calc_struct_size(self, format_string: str) -> int:
        """Calculate the size of the struct based on the format string."""
        if "Ls" in format_string:
            return 4  # Minimum size; actual size varies
        return struct.calcsize(format_string)

    def _pack_struct(self, values: Tuple[Any, ...]) -> bytes:
        """Pack values into binary data according to the format string."""
        if "Ls" in self.format_string:
            parts = self.format_string.split("Ls")
            fixed_part = parts[0] + (parts[1] if len(parts) > 1 else "")
            fixed_data = struct.pack(fixed_part, *values[:-1]) if fixed_part else b""
            string_val = (
                str(values[-1]).encode("utf-8") if values[-1] is not None else b""
            )
            string_len = len(string_val)
            return fixed_data + struct.pack("I", string_len) + string_val
        return struct.pack(self.format_string, *values)

    def _unpack_struct(self, data: bytes, offset: int) -> Tuple[Any, ...]:
        """Unpack binary data into values according to the format string."""
        if "Ls" in self.format_string:
            parts = self.format_string.split("Ls")
            fixed_part = parts[0] + (parts[1] if len(parts) > 1 else "")
            fixed_size = struct.calcsize(fixed_part)
            fixed_values = (
                struct.unpack(fixed_part, data[offset : offset + fixed_size])
                if fixed_part
                else ()
            )
            offset += fixed_size
            string_len = struct.unpack("I", data[offset : offset + 4])[0]
            offset += 4
            string_val = data[offset : offset + string_len].decode("utf-8")
            return fixed_values + (string_val,)
        struct_size = struct.calcsize(self.format_string)
        return struct.unpack(self.format_string, data[offset : offset + struct_size])

    def create(self, buffer: Gst.Buffer, values_list: List[Tuple[Any, ...]]) -> None:
        """
        Embed metadata into a GStreamer buffer.

        Args:
            buffer (Gst.Buffer): The GStreamer buffer to append metadata to.
            values_list (List[Tuple[Any, ...]]): A list of tuples containing values to pack.

        Raises:
            ValueError: If packing or compression fails.
        """
        try:
            struct_count = len(values_list)
            struct_data = b"".join(self._pack_struct(values) for values in values_list)

            name_length_bytes = struct.pack("I", self.name_length)
            count_bytes = struct.pack("I", struct_count)
            if self.include_format:
                format_length_bytes = struct.pack("I", self.format_length)
                core_data = (
                    name_length_bytes
                    + self.name_bytes
                    + format_length_bytes
                    + self.format_bytes
                    + count_bytes
                    + struct_data
                )
            else:
                core_data = (
                    name_length_bytes + self.name_bytes + count_bytes + struct_data
                )

            flags = 1 if self.compress else 0
            if self.compress:
                core_data = zlib.compress(core_data)
            else:
                core_data = self._pad_to(core_data)

            signature_part = self._pad_to(
                self.SIGNATURE + struct.pack("BB", self.VERSION, flags)
            )
            data_bytes = signature_part + core_data

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
        except zlib.error as e:
            raise ValueError(f"Compression failed: {str(e)}") from e

    def read(self, buffer: Gst.Buffer) -> List[Tuple[Any, ...]]:
        """
        Read metadata from a GStreamer buffer.

        Args:
            buffer (Gst.Buffer): The GStreamer buffer containing metadata.

        Returns:
            List[Tuple[Any, ...]]: A list of tuples containing the unpacked values.

        Raises:
            ValueError: If the metadata is invalid, corrupted, or cannot be decompressed.
        """
        n_mem = buffer.n_memory()
        if n_mem < 1:
            raise ValueError(f"No memory chunks found in buffer (got {n_mem})")

        memory = buffer.peek_memory(n_mem - 1)
        with memory.map(Gst.MapFlags.READ) as map_info:
            data_bytes = bytes(map_info.data)
            if len(data_bytes) < len(self.SIGNATURE) + 2:
                raise ValueError("Memory too short to contain signature and header")

            if data_bytes[: len(self.SIGNATURE)] != self.SIGNATURE:
                raise ValueError("Invalid signature in metadata")

            offset = len(self.SIGNATURE)
            version, flags = struct.unpack("BB", data_bytes[offset : offset + 2])
            if version != self.VERSION:
                raise ValueError(f"Unsupported version: {version}")
            is_compressed = flags & 1
            offset = self.signature_size

            core_data = data_bytes[offset:]
            if is_compressed:
                try:
                    core_data = zlib.decompress(core_data)
                except zlib.error as e:
                    raise ValueError(f"Decompression failed: {str(e)}") from e

            if len(core_data) < self.NAME_LENGTH_SIZE:
                raise ValueError("Memory too short to contain name length")
            name_length = struct.unpack("I", core_data[: self.NAME_LENGTH_SIZE])[0]
            offset = self.NAME_LENGTH_SIZE
            if len(core_data) < offset + name_length:
                raise ValueError("Memory too short to contain name")
            name = core_data[offset : offset + name_length].decode("utf-8")
            if name != self.name:
                raise ValueError(f"Name mismatch: expected {self.name}, got {name}")
            offset += name_length

            if self.include_format:
                if len(core_data) < offset + self.FORMAT_LENGTH_SIZE:
                    raise ValueError("Memory too short to contain format string length")
                format_length = struct.unpack(
                    "I", core_data[offset : offset + self.FORMAT_LENGTH_SIZE]
                )[0]
                offset += self.FORMAT_LENGTH_SIZE
                if len(core_data) < offset + format_length:
                    raise ValueError("Memory too short to contain format string")
                format_string = core_data[offset : offset + format_length].decode(
                    "utf-8"
                )
                offset += format_length
                if format_string != self.format_string:
                    raise ValueError(
                        f"Format string mismatch: expected {self.format_string}, got {format_string}"
                    )
            else:
                format_string = self.format_string

            if len(core_data) < offset + self.COUNT_SIZE:
                raise ValueError("Memory too short to contain struct count")
            struct_count = struct.unpack(
                "I", core_data[offset : offset + self.COUNT_SIZE]
            )[0]
            offset += self.COUNT_SIZE

            result = []
            for _ in range(struct_count):
                values = self._unpack_struct(core_data, offset)
                result.append(values)
                offset += self._calc_struct_size(format_string) + (
                    len(values[-1].encode("utf-8")) if "Ls" in format_string else 0
                )
            return result
