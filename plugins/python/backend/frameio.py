# FrameIO
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

"""Framework-agnostic interface for frame buffer I/O.

Video transform elements read pixel data out of a buffer as a numpy frame, run
inference, and either write a frame back or append an opaque metadata blob.
GStreamer expresses these via buffer mapping (`Gst.Buffer.map`), the muxed
buffer processor, and `append_memory`. Elements go through this interface
instead of touching those directly, so the per-buffer task logic that produces
numpy frames / metadata stays framework-free.

The `target` and `source` arguments are opaque handles (a Gst buffer and the
sink pad on the gst backend); callers only pass them through.
"""

from abc import ABC, abstractmethod


class FrameIO(ABC):
    """Read frames from and write frames/metadata back to a buffer/frame."""

    @abstractmethod
    def read_frames(self, target, source, width, height, framerate=(30, 1)):
        """Extract frame(s) from `target`. Returns a tuple
        ``(frames, num_sources, fmt)`` where ``frames`` is a numpy array shaped
        (H, W, C) for a single source or (N, H, W, C) for a batch (or None on
        failure), ``num_sources`` is the source count, and ``fmt`` is the pixel
        format string (e.g. "RGB")."""

    @abstractmethod
    def read_frame(self, target, source, width, height):
        """Read a single (H, W, C) RGB frame from `target`, or None on failure.
        Use this for plain (non-muxed) video buffers; use `read_frames` when the
        buffer may carry batched/muxed sources."""

    @abstractmethod
    def write_frame(self, target, frame):
        """Write an (H, W, C) uint8 numpy ``frame`` back into ``target`` in
        place. Returns True on success."""

    @abstractmethod
    def append_blob(self, target, header, payload):
        """Append an opaque metadata blob (``header`` bytes followed by
        ``payload`` bytes) to ``target``. Returns True on success."""

    def write_result(self, target, output=None, blob=None, header=None):
        """Write back what an element produced: a frame, a blob, or both.

        The tail nearly every video element shares. Either half may be absent:
        an element that only annotates returns no frame, one that only redraws
        returns no blob.
        """
        if output is not None:
            self.write_frame(target, output)
        if blob is not None:
            self.append_blob(target, header, blob)
