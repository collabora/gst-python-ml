# blob reader
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

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

BLOB_PREFIX = b"GST-"
BLOB_HEADER_END = b":"


def read_blobs(buffer, first_memory=1):
    # memory 0 is the video frame
    blobs = {}
    for index in range(first_memory, buffer.n_memory()):
        with buffer.peek_memory(index).map(Gst.MapFlags.READ) as info:
            if bytes(info.data[: len(BLOB_PREFIX)]) != BLOB_PREFIX:
                continue
            data = bytes(info.data)
        header, separator, payload = data.partition(BLOB_HEADER_END)
        if not separator:
            continue
        blobs[header[len(BLOB_PREFIX) :].decode().lower()] = payload
    return blobs
