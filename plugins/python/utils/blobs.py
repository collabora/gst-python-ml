# blob reader
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

BLOB_PREFIX = b"GST-"
BLOB_HEADER_END = b":"


def blob_name(header):
    return header[len(BLOB_PREFIX) :].partition(BLOB_HEADER_END)[0].decode().lower()


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
        blobs[blob_name(header)] = payload
    return blobs
