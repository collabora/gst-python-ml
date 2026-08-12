# Embedding
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
    from video_transform import VideoTransform
    from engine.embedding_engine import EmbeddingEngine
    from engine.engine_factory import EngineFactory
    from backend import frameio, GObject
    from tasks.embedding import EmbeddingTask

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(f"The 'embedding' element will not be available. Error {e}")

# Header prefix for embedding buffer metadata
EMBEDDING_META_HEADER = b"GST-EMBEDDING:"


class EmbeddingTransform(VideoTransform, EmbeddingTask):
    """
    GStreamer element for extracting frame embeddings for similarity search,
    clustering, or RAG.

    For each processed frame, extracts an embedding vector and appends it as a
    GST-EMBEDDING: memory chunk on the buffer. The chunk contains a JSON header
    followed by the raw float32 embedding bytes.

    Downstream elements can read the embedding:
      for i in range(buf.n_memory()):
          data = bytes(buf.peek_memory(i).map(Gst.MapFlags.READ).data)
          if data.startswith(b"GST-EMBEDDING:"):
              payload = data[14:]
              header_len = int.from_bytes(payload[:4], "little")
              header = json.loads(payload[4:4+header_len])
              dim = header["dim"]
              emb = np.frombuffer(payload[4+header_len:], dtype=np.float32)

    When a text property is set (CLIP only), a cosine similarity score between
    the text and image embeddings is also included in the JSON header.

    Use frame-stride to control processing frequency:
      pyml_embedding model-name=openai/clip-vit-large-patch14 frame-stride=5
    """

    __gstmetadata__ = (
        "Embedding Extractor",
        "Transform",
        "Extract frame embeddings using CLIP or DINOv2 for similarity search and RAG",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    normalize = GObject.Property(
        type=bool,
        default=True,
        nick="Normalize",
        blurb="L2-normalize the embedding vector",
        flags=GObject.ParamFlags.READWRITE,
    )

    text = GObject.Property(
        type=str,
        default=None,
        nick="Text Query",
        blurb="Optional text for computing text-image similarity (CLIP only)",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        self.mgr.engine_name = "pyml_embedding_engine"
        EngineFactory.register(self.mgr.engine_name, EmbeddingEngine)
        self._frame_count = 0
        self._text_embedding = None
        self._cached_text = None

    @GObject.Property(type=str)
    def engine_name(self):
        """Machine Learning Engine (read-only for this element)."""
        return self.mgr.engine_name

    @engine_name.setter
    def engine_name(self, value):
        raise ValueError("'engine_name' is read-only for pyml_embedding")

    @GObject.Property(type=int, default=0)
    def output_dim(self):
        """Embedding dimensionality (read-only, set after model load)."""
        if self.engine:
            return self.engine.output_dim
        return 0

    @output_dim.setter
    def output_dim(self, value):
        raise ValueError("'output_dim' is read-only")

    def _update_text_embedding(self):
        """Recompute cached text embedding when the text property changes."""
        if self.engine is None:
            return
        if self.text and self.text != self._cached_text:
            self._text_embedding = self.engine.do_text_embedding(
                self.text, normalize=self.normalize
            )
            self._cached_text = self.text
        elif not self.text:
            self._text_embedding = None
            self._cached_text = None

    def process_frames(self, frames, num_sources, fmt, target):
        """Embed the frame and append the payload, skipping strided-out frames."""
        self._frame_count += 1
        if self.frame_stride > 1 and (self._frame_count % self.frame_stride) != 1:
            return

        if self.engine is None:
            return

        frame = frames[0] if num_sources > 1 else frames

        emb = self.forward(frame)
        if emb is None:
            return

        # Update text embedding if needed
        self._update_text_embedding()

        # Portable task: serialize the length-prefixed embedding payload.
        _, payload = self.decode(emb)

        # Append embedding as a custom buffer memory chunk.
        # Format: HEADER_PREFIX + 4-byte header length + JSON header + raw float32
        frameio.write_result(target, None, payload, EMBEDDING_META_HEADER)

        self.logger.debug(f"Embedding extracted: dim={emb.shape[0]}")


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element(
        "pyml_embedding", EmbeddingTransform
    )
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'pyml_embedding' element will not be registered because required modules are missing."
    )
