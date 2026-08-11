# G2gFrameIO (g2g backend)
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Library General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.

"""Frame buffer I/O backed by the g2g host's `FrameBuffer`.

The g2g host hands each frame to `g2g_process(buf, w, h, fmt, sink)` where `buf`
is a `FrameBuffer`: a writable buffer-protocol view straight onto the frame's
system memory (no copy in or out). This backend reads it as a numpy frame, writes
a processed frame back in place, and routes opaque blobs to the host's `MetaSink`.

Unlike the GStreamer backend there is no muxed/batched buffer: the g2g host
delivers one source per `FrameBuffer` (batching is the aggregator's job, via
`g2g_process_batch`), so `read_frames` always reports a single source.
"""

import numpy as np

from backend.frameio import FrameIO

# Channel count per pixel-format string (the formats VideoTransform negotiates).
_CHANNELS = {
    "RGB": 3,
    "BGR": 3,
    "RGBA": 4,
    "ARGB": 4,
    "BGRA": 4,
    "ABGR": 4,
    "GRAY8": 1,
}


class G2gFrameIO(FrameIO):
    """`FrameIO` over the g2g `FrameBuffer` (pixels) and `MetaSink` (blobs)."""

    def __init__(self):
        # The per-frame MetaSink, bound by the element at the top of each
        # g2g_process call so append_blob has somewhere to put side-data.
        self._sink = None
        self._fmt = "RGB"

    def bind(self, sink, fmt="RGB"):
        """Bind the current frame's sink and pixel format (called per frame by
        the g2g element bases before any read/write)."""
        self._sink = sink
        self._fmt = (fmt or "RGB").upper()

    def _channels(self, fmt=None):
        return _CHANNELS.get((fmt or self._fmt).upper(), 3)

    def read_frame(self, target, source, width, height):
        c = self._channels()
        arr = np.frombuffer(target, dtype=np.uint8)
        if arr.size < width * height * c:
            return None
        return arr[: width * height * c].reshape((height, width, c))

    def read_frames(self, target, source, width, height, framerate=(30, 1)):
        # One source per FrameBuffer; report (frame, num_sources=1, fmt).
        frame = self.read_frame(target, source, width, height)
        if frame is None:
            return None, 0, self._fmt
        return frame, 1, self._fmt

    def write_frame(self, target, frame):
        # The FrameBuffer is writable, so frombuffer yields a writable view we
        # overwrite in place (no copy back to the host).
        view = np.frombuffer(target, dtype=np.uint8)
        flat = np.ascontiguousarray(frame, dtype=np.uint8).reshape(-1)
        n = min(view.size, flat.size)
        view[:n] = flat[:n]
        return True

    def append_blob(self, target, header, payload):
        if self._sink is None:
            return False
        hdr = header if isinstance(header, str) else bytes(header).decode("latin-1")
        self._sink.add_blob(hdr, bytes(payload))
        return True


#: The frame I/O implementation for this backend (shared singleton; the element
#: bases bind the per-frame sink on it, and leaves use it via `from backend import
#: frameio`). Defined here (not in the package __init__) so the element bases can
#: import it without a circular import through `backend`.
frameio = G2gFrameIO()
