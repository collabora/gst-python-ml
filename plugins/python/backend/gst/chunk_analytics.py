# ChunkAnalytics
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

import json
from collections import namedtuple

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

from backend.analytics import AnalyticsBackend  # noqa: E402
from utils.metadata import first_memory_with_prefix  # noqa: E402

ANALYTICS_CHUNK_HEADER = b"GST-ANALYTICS:"
OBJECT_FIELDS = ("label", "x", "y", "w", "h", "score")

ChunkEntry = namedtuple("ChunkEntry", "id")


class AnalyticsChunk:
    def __init__(self, buffer, document):
        self.buffer = buffer
        self.objects = document["objects"]
        self.classifications = document["classifications"]
        self.tracking = document["tracking"]
        self.relations = document["relations"]

    def length(self):
        return len(self.objects) + len(self.classifications) + len(self.tracking)

    def document(self):
        return {
            "objects": self.objects,
            "classifications": self.classifications,
            "tracking": self.tracking,
            "relations": self.relations,
        }


def write_chunk(chunk):
    payload = ANALYTICS_CHUNK_HEADER + json.dumps(chunk.document()).encode()
    # Gst.Memory.new_wrapped aborts in pygobject
    memory = Gst.Buffer.new_wrapped(payload).get_memory(0)
    index = first_memory_with_prefix(chunk.buffer, ANALYTICS_CHUNK_HEADER)
    if index is None:
        chunk.buffer.append_memory(memory)
        return
    chunk.buffer.replace_memory(index, memory)


class ChunkAnalytics(AnalyticsBackend):
    def add_relation_meta(self, buf):
        chunk = self.get_relation_meta(buf)
        if chunk is not None:
            return chunk
        chunk = AnalyticsChunk(
            buf, {"objects": [], "classifications": [], "tracking": [], "relations": []}
        )
        write_chunk(chunk)
        return chunk

    def get_relation_meta(self, buf):
        index = first_memory_with_prefix(buf, ANALYTICS_CHUNK_HEADER)
        if index is None:
            return None
        with buf.peek_memory(index).map(Gst.MapFlags.READ) as info:
            payload = bytes(info.data[len(ANALYTICS_CHUNK_HEADER) :])
        return AnalyticsChunk(buf, json.loads(payload))

    def remove_relation_meta(self, buf):
        index = first_memory_with_prefix(buf, ANALYTICS_CHUNK_HEADER)
        if index is None:
            return False
        buf.remove_memory(index)
        return True

    def relation_length(self, meta):
        return meta.length()

    def quark(self, label):
        return label

    def add_object(self, meta, label, x, y, w, h, score):
        entry = ChunkEntry(meta.length())
        # the relation meta stores integer boxes
        meta.objects.append(
            {
                "id": entry.id,
                "label": label,
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "score": float(score),
            }
        )
        write_chunk(meta)
        return entry

    def add_classification(self, meta, index, label):
        entry = ChunkEntry(meta.length())
        meta.classifications.append({"id": entry.id, "index": index, "label": label})
        write_chunk(meta)
        return entry

    def add_tracking(self, meta, track_id, timestamp=None):
        if timestamp is None:
            timestamp = Gst.util_get_timestamp()
        entry = ChunkEntry(meta.length())
        meta.tracking.append(
            {"id": entry.id, "track_id": track_id, "timestamp": timestamp}
        )
        write_chunk(meta)
        return entry

    def relate(self, meta, src, dst):
        meta.relations.append([src.id, dst.id])
        write_chunk(meta)
        return True

    def read_objects(self, meta):
        return [{field: obj[field] for field in OBJECT_FIELDS} for obj in meta.objects]
