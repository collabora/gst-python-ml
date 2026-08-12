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

import threading

import numpy as np

from backend.frameio import FrameIO

#: For each pixel format the host carries: how many channels a pixel takes, and
#: which of them are R, G and B. `None` for a format with no colour to pick out.
_FORMATS = {
    "RGB": (3, (0, 1, 2)),
    "BGR": (3, (2, 1, 0)),
    "RGBA": (4, (0, 1, 2)),
    "ARGB": (4, (1, 2, 3)),
    "BGRA": (4, (2, 1, 0)),
    "ABGR": (4, (3, 2, 1)),
    "GRAY8": (1, None),
}


def as_rgb(frames, fmt):
    """The RGB view of `frames` the ML elements infer over.

    Nothing converts pixels ahead of a hosted element the way `videoconvert`
    does in a gst pipeline, so a host format that is not already RGB is reduced
    here. Read-only: the write-back target is still the original buffer.
    """
    channels, rgb = _FORMATS.get((fmt or "RGB").upper(), _FORMATS["RGB"])
    already_rgb = channels == 3 and rgb == (0, 1, 2)
    if rgb is None or already_rgb:
        return frames
    return frames[..., list(rgb)]


class G2gFrameIO(FrameIO):
    """`FrameIO` over the g2g `FrameBuffer` (pixels) and `MetaSink` (blobs)."""

    def __init__(self):
        # Per-thread because the host runs one thread per element: one shared
        # binding would send an element's blobs to whichever element bound last.
        self._bound = threading.local()

    def bind(self, sink, fmt="RGB"):
        """Bind the current frame's sink and pixel format (called per frame by
        the g2g element bases before any read/write)."""
        self._bound.sink = sink
        self._bound.fmt = (fmt or "RGB").upper()

    @property
    def _sink(self):
        return getattr(self._bound, "sink", None)

    @property
    def _fmt(self):
        return getattr(self._bound, "fmt", "RGB")

    def _channels(self, fmt=None):
        channels, _ = _FORMATS.get((fmt or self._fmt).upper(), _FORMATS["RGB"])
        return channels

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


#: Defined here rather than in the package __init__ so the element bases can
#: import it without a circular import through `backend`.
frameio = G2gFrameIO()
