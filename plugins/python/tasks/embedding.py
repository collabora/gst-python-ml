# EmbeddingTask
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

"""Backend-agnostic embedding task.

`EmbeddingTask` runs the embedding forward pass and serializes the result into
the length-prefixed payload that follows the GST-EMBEDDING: header. It uses only
numpy/json/struct and the engine; it never touches the buffer. The backend
element shell does the frame read and appends the payload via the frameio
facade, prepending the header.

Contract expected from the host element:
  self.engine          - the ML engine (provided by MLEngineMixin)
  self.normalize       - L2-normalize flag (a backend-declared property)
  self.text            - optional text query (a backend-declared property)
  self._text_embedding - cached text embedding (host-managed state)
"""

import json
import struct


class EmbeddingTask:
    """Inference + payload serialization, independent of any framework."""

    def forward(self, frame):
        return self.task_engine.do_forward(frame, normalize=self.normalize)

    def decode(self, emb):
        """Serialize an embedding into ``(None, payload_bytes)``.

        ``payload_bytes`` is everything AFTER the GST-EMBEDDING: header: the
        4-byte JSON header length, the JSON header, and the raw float32 bytes.
        The shell prepends the header when appending the blob.
        """
        import numpy as np

        # Build JSON header with dimension info and optional similarity score
        header = {"dim": int(emb.shape[0]), "dtype": "float32"}
        if self._text_embedding is not None:
            similarity = float(np.dot(emb, self._text_embedding))
            header["text"] = self.text
            header["similarity"] = round(similarity, 6)

        header_bytes = json.dumps(header).encode("utf-8")
        header_len = struct.pack("<I", len(header_bytes))
        emb_bytes = emb.tobytes()

        # Format (after the header prefix): 4-byte header length + JSON header
        # + raw float32 embedding bytes.
        payload = header_len + header_bytes + emb_bytes
        return None, payload
