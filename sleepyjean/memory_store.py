"""Episode and abstract memory store for SleepyJean.

Uses ChromaDB for vector search and SQLite for metadata tracking.
Separate from shared_state's ChromaDB collection (different collection name).
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from datetime import datetime
from typing import Any

import chromadb
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_ISO_FMT = "%Y-%m-%dT%H:%M:%S.%f"


class MemoryStore:
    """Episode memory + abstract memory (cluster centroids) store."""

    def __init__(self, config: dict) -> None:
        collection_name = config.get("chromadb_collection", "sleepyjean_memory")
        sqlite_path = config.get("sqlite_path", "data/sleepyjean_memory.db")
        persist_dir = config.get("chromadb_persist_directory")

        # ChromaDB
        if persist_dir:
            self._client = chromadb.PersistentClient(path=persist_dir)
        else:
            self._client = chromadb.EphemeralClient()

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # SQLite for metadata
        self._db = sqlite3.connect(sqlite_path)
        self._db.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS memory_metadata (
                episode_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                source TEXT,
                access_count INTEGER NOT NULL DEFAULT 0,
                confidence REAL NOT NULL DEFAULT 1.0,
                is_abstract INTEGER NOT NULL DEFAULT 0,
                cluster_id TEXT
            )
        """)
        self._db.commit()

    def add_episode(
        self,
        embedding: NDArray[np.float32],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add an episode memory. Returns episode_id."""
        metadata = metadata or {}
        episode_id = str(uuid.uuid4())
        now = datetime.now().strftime(_ISO_FMT)

        self._collection.add(
            ids=[episode_id],
            embeddings=[embedding.tolist()],
            metadatas=[{"is_abstract": False, "created_at": now}],
        )

        self._db.execute(
            """INSERT INTO memory_metadata
               (episode_id, created_at, source, access_count, confidence, is_abstract)
               VALUES (?, ?, ?, 0, ?, 0)""",
            (episode_id, now, metadata.get("source", ""), metadata.get("confidence", 1.0)),
        )
        self._db.commit()

        return episode_id

    def add_abstract(
        self,
        embedding: NDArray[np.float32],
        cluster_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add a cluster centroid as abstract memory."""
        metadata = metadata or {}
        episode_id = str(uuid.uuid4())
        now = datetime.now().strftime(_ISO_FMT)

        self._collection.add(
            ids=[episode_id],
            embeddings=[embedding.tolist()],
            metadatas=[{"is_abstract": True, "created_at": now, "cluster_id": cluster_id}],
        )

        self._db.execute(
            """INSERT INTO memory_metadata
               (episode_id, created_at, source, access_count, confidence, is_abstract, cluster_id)
               VALUES (?, ?, ?, 0, ?, 1, ?)""",
            (episode_id, now, metadata.get("source", ""), metadata.get("confidence", 1.0), cluster_id),
        )
        self._db.commit()

        return episode_id

    def search(
        self,
        query_embedding: NDArray[np.float32],
        n_results: int = 3,
        include_abstract: bool = True,
    ) -> list[dict]:
        """Cosine similarity search. Returns list of {episode_id, embedding, distance, metadata}.

        Increments access_count for each result.
        """
        total = self._collection.count()
        if total == 0:
            return []

        # ChromaDB where filter for abstract exclusion
        where = None
        if not include_abstract:
            where = {"is_abstract": False}

        effective_n = min(n_results, total)
        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=effective_n,
            where=where,
            include=["embeddings", "distances", "metadatas"],
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        out = []
        ids_to_update = []
        for i, eid in enumerate(results["ids"][0]):
            out.append({
                "episode_id": eid,
                "embedding": np.array(results["embeddings"][0][i], dtype=np.float32),
                "distance": results["distances"][0][i],
                "metadata": results["metadatas"][0][i],
            })
            ids_to_update.append(eid)

        # Increment access_count
        if ids_to_update:
            placeholders = ",".join("?" for _ in ids_to_update)
            self._db.execute(
                f"UPDATE memory_metadata SET access_count = access_count + 1 WHERE episode_id IN ({placeholders})",
                ids_to_update,
            )
            self._db.commit()

        return out

    def get_all_episodes(self) -> list[dict]:
        """Get all episode memories (not abstracts). For Reconsolidation."""
        total = self._collection.count()
        if total == 0:
            return []

        results = self._collection.get(
            where={"is_abstract": False},
            include=["embeddings", "metadatas"],
        )

        if not results["ids"]:
            return []

        out = []
        for i, eid in enumerate(results["ids"]):
            # Get SQLite metadata
            row = self._db.execute(
                "SELECT access_count, confidence FROM memory_metadata WHERE episode_id = ?",
                (eid,),
            ).fetchone()
            access_count = row["access_count"] if row else 0
            confidence = row["confidence"] if row else 1.0

            out.append({
                "episode_id": eid,
                "embedding": np.array(results["embeddings"][i], dtype=np.float32),
                "metadata": results["metadatas"][i],
                "access_count": access_count,
                "confidence": confidence,
            })

        return out

    def delete(self, episode_ids: list[str]) -> int:
        """Delete specified episodes. Returns number deleted."""
        if not episode_ids:
            return 0

        # Filter to existing ids
        existing = set(self._collection.get(ids=episode_ids)["ids"])
        to_delete = [eid for eid in episode_ids if eid in existing]

        if not to_delete:
            return 0

        self._collection.delete(ids=to_delete)

        placeholders = ",".join("?" for _ in to_delete)
        self._db.execute(
            f"DELETE FROM memory_metadata WHERE episode_id IN ({placeholders})",
            to_delete,
        )
        self._db.commit()

        return len(to_delete)

    def replace_abstracts(self, abstracts: list[dict]) -> None:
        """Replace all abstract memories with new cluster centroids.

        Each abstract: {embedding: NDArray, cluster_id: str, metadata: dict}
        """
        # Delete all existing abstracts
        existing = self._collection.get(
            where={"is_abstract": True},
            include=[],
        )
        if existing["ids"]:
            self._collection.delete(ids=existing["ids"])
            placeholders = ",".join("?" for _ in existing["ids"])
            self._db.execute(
                f"DELETE FROM memory_metadata WHERE episode_id IN ({placeholders})",
                existing["ids"],
            )

        # Add new abstracts
        for abstract in abstracts:
            self.add_abstract(
                embedding=abstract["embedding"],
                cluster_id=abstract["cluster_id"],
                metadata=abstract.get("metadata", {}),
            )

    def stats(self) -> dict:
        """Return memory statistics."""
        total = self._collection.count()

        row = self._db.execute(
            "SELECT COUNT(*) as cnt FROM memory_metadata WHERE is_abstract = 0"
        ).fetchone()
        episode_count = row["cnt"] if row else 0

        row = self._db.execute(
            "SELECT COUNT(*) as cnt FROM memory_metadata WHERE is_abstract = 1"
        ).fetchone()
        abstract_count = row["cnt"] if row else 0

        row = self._db.execute(
            "SELECT COUNT(DISTINCT cluster_id) as cnt FROM memory_metadata WHERE cluster_id IS NOT NULL"
        ).fetchone()
        cluster_count = row["cnt"] if row else 0

        row = self._db.execute(
            "SELECT MAX(created_at) as latest FROM memory_metadata"
        ).fetchone()
        last_updated = row["latest"] if row else None

        return {
            "total": total,
            "episode_count": episode_count,
            "abstract_count": abstract_count,
            "cluster_count": cluster_count,
            "last_updated": last_updated,
        }

    def close(self) -> None:
        """Close SQLite connection."""
        self._db.close()
