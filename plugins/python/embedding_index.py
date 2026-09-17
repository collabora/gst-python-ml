# EmbeddingIndex
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

import sqlite3

import numpy as np

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS embeddings (
    source_id TEXT NOT NULL,
    pts REAL NOT NULL,
    model_name TEXT NOT NULL,
    vector BLOB NOT NULL
)
"""
INSERT_ROW = (
    "INSERT INTO embeddings (source_id, pts, model_name, vector) VALUES (?, ?, ?, ?)"
)
SELECT_ROWS = "SELECT source_id, pts, vector FROM embeddings"
SELECT_MODEL_NAME = "SELECT model_name FROM embeddings LIMIT 1"

VECTOR_DTYPE = np.float32
SMALLEST_USABLE_NORM = 1e-12


class EmbeddingIndex:
    def __init__(self, connection):
        self.connection = connection

    @classmethod
    def open(cls, path):
        # a sink opens the index on the state change thread and writes on the streaming thread
        connection = sqlite3.connect(path, check_same_thread=False)
        connection.execute(CREATE_TABLE)
        connection.commit()
        return cls(connection)

    def close(self):
        self.connection.close()

    def model_name(self):
        row = self.connection.execute(SELECT_MODEL_NAME).fetchone()
        return row[0] if row else ""

    def add(self, source_id, pts_seconds, vector, model_name):
        held = self.model_name()
        if held and held != model_name:
            raise ValueError(
                f"the index holds {held} embeddings, it cannot also hold {model_name}"
            )
        blob = np.asarray(vector, dtype=VECTOR_DTYPE).tobytes()
        self.connection.execute(
            INSERT_ROW, (source_id, float(pts_seconds), model_name, blob)
        )
        self.connection.commit()

    def search(self, vector, count):
        rows = self.connection.execute(SELECT_ROWS).fetchall()
        if not rows:
            return []
        query = np.asarray(vector, dtype=VECTOR_DTYPE)
        stored = np.stack([np.frombuffer(row[2], dtype=VECTOR_DTYPE) for row in rows])
        norms = np.linalg.norm(stored, axis=1) * np.linalg.norm(query)
        scores = stored @ query / np.maximum(norms, SMALLEST_USABLE_NORM)
        best = np.argsort(-scores)[:count]
        return [
            {
                "pts": rows[index][1],
                "source_id": rows[index][0],
                "score": float(scores[index]),
            }
            for index in best
        ]
