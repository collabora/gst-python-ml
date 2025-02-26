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
from typing import Any, Tuple
import zlib
from gi.repository import Gst
from log.logger_factory import LoggerFactory


class Metadata:
    """A class to manage sophisticated metadata for GStreamer buffers."""

    SIGNATURE = b"GST-PYTHON-ML"  # Fixed signature for identification
    VERSION = 1  # Current metadata version
    FORMAT_LENGTH_SIZE = 4  # Size for format string length (uint32)
    ALIGNMENT = 4  # Alignment boundary in bytes

    def __init__(
        self, format_string: str, include_format: bool = True, compress: bool = False
    ):
        """
        Initialize the metadata handler.

        Args:
            format_string: The struct format string (e.g., "if" for int and float).
            include_format: Whether to include the format string in the metadata.
            compress: Whether to compress the payload with zlib.
        """
        self.format_string = format_string
        self.struct_size = struct.calcsize(format_string)
        self.logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)
        self.include_format = include_format
        self.compress = compress

        if include_format:
            self.format_bytes = format_string.encode("utf-8")
            self.format_length = len(self.format_bytes)
            self.core_size = (
                self.FORMAT_LENGTH_SIZE + self.format_length + self.struct_size
            )
        else:
            self.core_size = self.struct_size

        # Precompute aligned sizes
        self.signature_size = self._aligned_size(
            len(self.SIGNATURE) + 2
        )  # Signature + version + flags
        self.total_size = self.signature_size + (
            self._aligned_size(self.core_size) if not compress else self.core_size
        )

    def _aligned_size(self, size: int) -> int:
        """Calculate size with padding to the next alignment boundary."""
        return size + ((self.ALIGNMENT - (size % self.ALIGNMENT)) % self.ALIGNMENT)

    def _pad_to(self, data: bytes, alignment: int = ALIGNMENT) -> bytes:
        """Pad data to the specified alignment with null bytes."""
        padding = (alignment - (len(data) % alignment)) % alignment
        return data + b"\x00" * padding

    def create(self, buffer: Gst.Buffer, *values: Any) -> None:
        """
        Creates structured data and attaches it to the provided GstBuffer.
        Layout:
            - Signature ("GST-PYTHON-ML")
            - Version (1 byte)
            - Flags (1 byte: bit 0 = compression)
            - [Aligned padding]
            - Core data (optional format length + format string + struct data, possibly compressed)

        Args:
            buffer: GstBuffer to attach the memory to.
            *values: Values to pack according to format_string.

        Raises:
            ValueError: If values don't match the format string in count or type.
        """
        try:
            # Pack struct data
            struct_data = struct.pack(self.format_string, *values)

            # Prepare core data
            if self.include_format:
                format_length_bytes = struct.pack("I", self.format_length)
                core_data = format_length_bytes + self.format_bytes + struct_data
            else:
                core_data = struct_data

            # Compress if enabled
            flags = 1 if self.compress else 0
            if self.compress:
                core_data = zlib.compress(core_data)
            else:
                core_data = self._pad_to(core_data)  # Pad only if not compressed

            # Combine with signature, version, and flags
            signature_part = self._pad_to(
                self.SIGNATURE + struct.pack("BB", self.VERSION, flags)
            )
            data_bytes = signature_part + core_data

            # Create and attach memory
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

    def read(self, memory: Gst.Memory) -> Tuple[Any, ...]:
        """
        Reads structured data from memory.

        Args:
            memory: GstMemory object to read from.

        Returns:
            Tuple of unpacked values according to the format string.

        Raises:
            ValueError: If the memory layout is invalid, corrupted, or incompatible.
        """
        with memory.map(Gst.MapFlags.READ) as map_info:
            data_bytes = map_info.data
            if len(data_bytes) < len(self.SIGNATURE) + 2:
                raise ValueError("Memory too short to contain signature and header")

            # Check signature
            if data_bytes[: len(self.SIGNATURE)] != self.SIGNATURE:
                raise ValueError("Invalid signature in metadata")

            # Parse version and flags
            offset = len(self.SIGNATURE)
            version, flags = struct.unpack("BB", data_bytes[offset : offset + 2])
            if version != self.VERSION:
                raise ValueError(f"Unsupported version: {version}")
            is_compressed = flags & 1
            offset = self.signature_size  # Skip to aligned core data

            # Extract and decompress core data
            core_data = data_bytes[offset:]
            if is_compressed:
                try:
                    core_data = zlib.decompress(core_data)
                except zlib.error as e:
                    raise ValueError(f"Decompression failed: {str(e)}") from e

            # Parse core data
            if self.include_format:
                if len(core_data) < self.FORMAT_LENGTH_SIZE:
                    raise ValueError("Memory too short to contain format string length")
                format_length = struct.unpack(
                    "I", core_data[: self.FORMAT_LENGTH_SIZE]
                )[0]
                offset = self.FORMAT_LENGTH_SIZE
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
                offset = 0

            # Unpack struct data
            struct_size = struct.calcsize(format_string)
            if len(core_data) < offset + struct_size:
                raise ValueError(
                    f"Memory too short for struct data: expected {struct_size}, got {len(core_data) - offset}"
                )
            return struct.unpack(
                format_string, core_data[offset : offset + struct_size]
            )


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
