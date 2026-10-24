# EmbeddingTask
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

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
