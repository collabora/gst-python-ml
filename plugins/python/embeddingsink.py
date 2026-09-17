# EmbeddingSink
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

from log.global_logger import GlobalLogger
import backend

CAN_REGISTER_ELEMENT = True
try:
    import json

    import numpy as np

    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    from gi.repository import Gst, GstBase  # noqa: E402

    from log.logger_factory import LoggerFactory  # noqa: E402
    from backend import GObject  # noqa: E402
    from embedding_index import EmbeddingIndex  # noqa: E402
    from utils.blobs import read_blobs  # noqa: E402
except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'pyml_embeddingsink' element will not be available. Error: {e}"
    )

# the header pyml_embedding attaches, lowercased by read_blobs
EMBEDDING_BLOB = "embedding"
HEADER_LENGTH_BYTES = 4
HEADER_MODEL_KEY = "model_name"
VECTOR_DTYPE = "float32"


def decode_embedding(payload):
    header_length = int.from_bytes(payload[:HEADER_LENGTH_BYTES], "little")
    header_end = HEADER_LENGTH_BYTES + header_length
    header = json.loads(payload[HEADER_LENGTH_BYTES:header_end])
    return header, np.frombuffer(payload[header_end:], dtype=VECTOR_DTYPE)


class EmbeddingSink(GstBase.BaseSink):
    GST_PLUGIN_NAME = "pyml_embeddingsink"

    __gstmetadata__ = (
        "Embedding Sink",
        "Sink",
        "Stores the embedding vector of each buffer in a searchable sqlite index",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    if backend.BACKEND == "gst":
        __gsttemplates__ = Gst.PadTemplate.new(
            "sink",
            Gst.PadDirection.SINK,
            Gst.PadPresence.ALWAYS,
            Gst.Caps.new_any(),
        )

    location = GObject.Property(
        type=str,
        default="",
        nick="Location",
        blurb="sqlite file holding the embedding index",
    )

    source_id = GObject.Property(
        type=str,
        default="",
        nick="Source Id",
        blurb="Name a search result reports for the stream these embeddings come from",
    )

    model_name = GObject.Property(
        type=str,
        default="",
        nick="Model Name",
        blurb="Embedding model to record when the blob header does not name one",
    )

    def __init__(self):
        super().__init__()
        self.logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)
        self.set_sync(False)
        # a sporadic feed like a caption pad must not hold the pipeline in preroll
        self.set_async_enabled(False)
        self._index = None

    def do_start(self):
        if not self.location:
            self.logger.error("pyml_embeddingsink needs a location to write the index")
            return False
        self._index = EmbeddingIndex.open(self.location)
        return True

    def do_stop(self):
        self._index.close()
        self._index = None
        return True

    def do_render(self, buffer):
        payload = read_blobs(buffer).get(EMBEDDING_BLOB)
        if payload is None or buffer.pts == Gst.CLOCK_TIME_NONE:
            return Gst.FlowReturn.OK
        header, vector = decode_embedding(payload)
        model_name = header.get(HEADER_MODEL_KEY) or self.model_name
        if not model_name:
            self.logger.error(
                "the embedding blob names no model, set model-name on pyml_embeddingsink"
            )
            return Gst.FlowReturn.ERROR
        self._index.add(self.source_id, buffer.pts / Gst.SECOND, vector, model_name)
        return Gst.FlowReturn.OK


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element(
        EmbeddingSink.GST_PLUGIN_NAME, EmbeddingSink
    )
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'pyml_embeddingsink' element will not be registered because required modules are missing."
    )
