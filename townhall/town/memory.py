"""Long term memory across meeting rounds: SQLite plus sentence embeddings.

One row per (resident, topic). A newer round replaces the older opinion on the same topic; an
older round never overwrites a newer one. recall() returns a resident's top k memories by cosine
similarity to a query. delete_resident() and delete_all() keep the responsible use promise that
users can delete saved memories; secure_delete overwrites deleted rows inside the file.
"""

import json
import sqlite3
from pathlib import Path

import numpy as np

from townhall import config

DB_PATH = config.PROCESSED / "town" / "memory.sqlite"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Memory:
    def __init__(self, path=DB_PATH, embed=None):
        """embed: optional callable list[str] -> 2D array; defaults to all-MiniLM-L6-v2, loaded on first use."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA secure_delete = ON")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS memories (resident_id TEXT, topic TEXT, stance TEXT, reasons TEXT, "
            "comment TEXT, round INTEGER, embedding BLOB, PRIMARY KEY (resident_id, topic))"
        )
        self._embed = embed

    def embed(self, texts):
        if self._embed is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(MODEL)
            self._embed = lambda batch: model.encode(batch)
        vectors = np.asarray(self._embed(texts), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1, norms)  # unit length, so a dot product is cosine similarity

    def remember(self, resident_id, topic, stance, reasons, comment, round):
        """Save an opinion. Returns False when a newer round already holds this topic."""
        vector = self.embed([f"{topic}. {stance}. {' '.join(reasons)} {comment}"])[0]
        cursor = self.db.execute(
            "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT (resident_id, topic) DO UPDATE SET "
            "stance = excluded.stance, reasons = excluded.reasons, comment = excluded.comment, "
            "round = excluded.round, embedding = excluded.embedding WHERE excluded.round >= memories.round",
            (resident_id, topic, stance, json.dumps(list(reasons)), comment, int(round), vector.tobytes()),
        )
        self.db.commit()
        return cursor.rowcount == 1

    def recall(self, resident_id, query, k=3):
        """A resident's k memories most similar to query, best first, each with its score."""
        rows = self.db.execute(
            "SELECT topic, stance, reasons, comment, round, embedding FROM memories WHERE resident_id = ?",
            (resident_id,),
        ).fetchall()
        if not rows:
            return []
        scores = np.stack([np.frombuffer(row[5], dtype=np.float32) for row in rows]) @ self.embed([query])[0]
        best = np.argsort(-scores)[:k]
        return [{"topic": rows[i][0], "stance": rows[i][1], "reasons": json.loads(rows[i][2]),
                 "comment": rows[i][3], "round": rows[i][4], "score": float(scores[i])} for i in best]

    def count(self, resident_id=None):
        if resident_id is None:
            return self.db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        return self.db.execute("SELECT COUNT(*) FROM memories WHERE resident_id = ?", (resident_id,)).fetchone()[0]

    def delete_resident(self, resident_id):
        """Delete every memory of one resident. Returns how many rows were deleted."""
        deleted = self.db.execute("DELETE FROM memories WHERE resident_id = ?", (resident_id,)).rowcount
        self.db.commit()
        return deleted

    def delete_all(self):
        deleted = self.db.execute("DELETE FROM memories").rowcount
        self.db.commit()
        self.db.execute("VACUUM")  # shrink the file so deleted text is not left in free pages
        return deleted

    def close(self):
        self.db.close()
