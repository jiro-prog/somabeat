"""Tests for Phase 4 excretory system — SleepyJean data cleanup (T38)."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import chromadb
import pytest

from bridge.cleaner import (
    clean_chromadb_rag,
    clean_lora_adapters,
    _find_duplicate_entries,
    _find_aged_entries,
)


# ============================================================
# Helper: seed a ChromaDB collection with test entries
# ============================================================


def _seed_collection(tmpdir: str, entries: list[dict]) -> str:
    """Create a ChromaDB collection with the given entries.

    Each entry: {"id": str, "topic": str, "date": str, "source_type": str, "text": str}
    Returns the chromadb_path.
    """
    db_path = str(Path(tmpdir) / "chromadb")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection("sleepyjean_knowledge")

    for e in entries:
        collection.add(
            ids=[e["id"]],
            documents=[e.get("text", f"doc for {e['id']}")],
            metadatas=[{
                "topic": e.get("topic", "unknown"),
                "date": e.get("date", "2026-01-01"),
                "source_type": e.get("source_type", "exploration"),
            }],
        )

    return db_path


def _get_remaining_ids(db_path: str) -> set[str]:
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection("sleepyjean_knowledge")
    return set(collection.get()["ids"])


# ============================================================
# ChromaDB cleanup tests
# ============================================================


class TestCleanChromaDBRag:
    def test_single_topic_single_generation(self):
        """Only generation for a topic → kept."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = _seed_collection(tmpdir, [
                {"id": "2026-03-15_asyncio_0_exploration", "topic": "asyncio", "date": "2026-03-15"},
                {"id": "2026-03-15_asyncio_1_qa", "topic": "asyncio", "date": "2026-03-15"},
            ])
            result = clean_chromadb_rag(db_path)
            assert result["duplicate_removed"] == 0
            assert result["aged_removed"] == 0
            assert result["retained"] == 2

    def test_single_topic_multiple_generations(self):
        """Older generation removed, latest kept."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = _seed_collection(tmpdir, [
                # Old generation
                {"id": "2026-03-14_asyncio_0_exploration", "topic": "asyncio", "date": "2026-03-14"},
                {"id": "2026-03-14_asyncio_1_qa", "topic": "asyncio", "date": "2026-03-14"},
                # New generation
                {"id": "2026-03-15_asyncio_0_exploration", "topic": "asyncio", "date": "2026-03-15"},
                {"id": "2026-03-15_asyncio_1_qa", "topic": "asyncio", "date": "2026-03-15"},
            ])
            result = clean_chromadb_rag(db_path)
            assert result["duplicate_removed"] == 2
            assert result["retained"] == 2

            remaining = _get_remaining_ids(db_path)
            assert "2026-03-15_asyncio_0_exploration" in remaining
            assert "2026-03-15_asyncio_1_qa" in remaining
            assert "2026-03-14_asyncio_0_exploration" not in remaining

    def test_multiple_topics(self):
        """Topics are processed independently."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = _seed_collection(tmpdir, [
                # asyncio: 2 generations
                {"id": "2026-03-14_asyncio_0", "topic": "asyncio", "date": "2026-03-14"},
                {"id": "2026-03-15_asyncio_0", "topic": "asyncio", "date": "2026-03-15"},
                # quantum: 1 generation
                {"id": "2026-03-14_quantum_0", "topic": "quantum_mechanics", "date": "2026-03-14"},
            ])
            result = clean_chromadb_rag(db_path)
            assert result["duplicate_removed"] == 1  # only asyncio old gen
            assert result["retained"] == 2

            remaining = _get_remaining_ids(db_path)
            assert "2026-03-15_asyncio_0" in remaining
            assert "2026-03-14_quantum_0" in remaining
            assert "2026-03-14_asyncio_0" not in remaining

    def test_max_generations_2(self):
        """max_generations=2 keeps 2 date-generations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = _seed_collection(tmpdir, [
                {"id": "d1", "topic": "ai", "date": "2026-03-13"},
                {"id": "d2", "topic": "ai", "date": "2026-03-14"},
                {"id": "d3", "topic": "ai", "date": "2026-03-15"},
            ])
            result = clean_chromadb_rag(db_path, max_generations_per_topic=2)
            assert result["duplicate_removed"] == 1
            assert result["retained"] == 2

            remaining = _get_remaining_ids(db_path)
            assert "d1" not in remaining  # oldest removed
            assert "d2" in remaining
            assert "d3" in remaining

    def test_aged_fallback(self):
        """Entries older than max_age_days are removed after dedup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_date = (datetime.now() - timedelta(days=100)).strftime("%Y-%m-%d")
            recent_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")

            db_path = _seed_collection(tmpdir, [
                {"id": "old_topic_0", "topic": "old_topic", "date": old_date},
                {"id": "recent_topic_0", "topic": "recent_topic", "date": recent_date},
            ])
            result = clean_chromadb_rag(db_path, max_age_days=90)
            assert result["duplicate_removed"] == 0
            assert result["aged_removed"] == 1
            assert result["retained"] == 1

            remaining = _get_remaining_ids(db_path)
            assert "recent_topic_0" in remaining
            assert "old_topic_0" not in remaining

    def test_aged_recent_kept(self):
        """Entries within max_age_days are kept."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recent_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

            db_path = _seed_collection(tmpdir, [
                {"id": "recent_0", "topic": "topic_a", "date": recent_date},
            ])
            result = clean_chromadb_rag(db_path, max_age_days=90)
            assert result["aged_removed"] == 0
            assert result["retained"] == 1

    def test_empty_collection(self):
        """Empty collection → no errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "chromadb")
            client = chromadb.PersistentClient(path=db_path)
            client.get_or_create_collection("sleepyjean_knowledge")

            result = clean_chromadb_rag(db_path)
            assert result["duplicate_removed"] == 0
            assert result["aged_removed"] == 0
            assert result["retained"] == 0

    def test_no_metadata_topic(self):
        """Missing topic in metadata → grouped as 'unknown'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "chromadb")
            client = chromadb.PersistentClient(path=db_path)
            col = client.get_or_create_collection("sleepyjean_knowledge")

            # Add entries with missing topic
            col.add(
                ids=["no_topic_1", "no_topic_2"],
                documents=["doc1", "doc2"],
                metadatas=[
                    {"date": "2026-03-14"},  # no topic
                    {"date": "2026-03-15"},  # no topic
                ],
            )

            result = clean_chromadb_rag(db_path)
            # Both fall under "unknown" topic. Two dates → old gen removed.
            assert result["duplicate_removed"] == 1
            assert result["retained"] == 1

    def test_collection_not_found(self):
        """Non-existent collection → graceful return."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "chromadb")
            client = chromadb.PersistentClient(path=db_path)
            # Don't create the collection

            result = clean_chromadb_rag(db_path, collection_name="nonexistent")
            assert result["duplicate_removed"] == 0
            assert result["retained"] == 0


# ============================================================
# LoRA adapter cleanup tests
# ============================================================


class TestCleanLoraAdapters:
    def test_basic(self):
        """Old adapters removed, newest max_keep kept."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "lora"
            adapter_dir.mkdir()

            for name in ["lora_2026-03-10", "lora_2026-03-11", "lora_2026-03-12",
                         "lora_2026-03-13", "lora_2026-03-14"]:
                (adapter_dir / name).mkdir()
                (adapter_dir / name / "adapter_model.bin").write_text("fake")

            result = clean_lora_adapters(str(adapter_dir), max_keep=3)
            assert result["removed"] == 2
            assert result["retained"] == 3

            remaining = {d.name for d in adapter_dir.iterdir() if d.is_dir()}
            assert "lora_2026-03-12" in remaining
            assert "lora_2026-03-13" in remaining
            assert "lora_2026-03-14" in remaining
            assert "lora_2026-03-10" not in remaining
            assert "lora_2026-03-11" not in remaining

    def test_current_lora_protected(self):
        """current_lora symlink target is always protected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "lora"
            adapter_dir.mkdir()

            for name in ["lora_2026-03-10", "lora_2026-03-11", "lora_2026-03-12",
                         "lora_2026-03-13", "lora_2026-03-14"]:
                (adapter_dir / name).mkdir()
                (adapter_dir / name / "adapter_model.bin").write_text("fake")

            # current_lora points to the oldest adapter
            (adapter_dir / "current_lora").symlink_to(adapter_dir / "lora_2026-03-10")

            result = clean_lora_adapters(str(adapter_dir), max_keep=3)

            remaining = {d.name for d in adapter_dir.iterdir() if d.is_dir()}
            # lora_2026-03-10 protected by current_lora
            assert "lora_2026-03-10" in remaining
            # newest 3 kept
            assert "lora_2026-03-12" in remaining
            assert "lora_2026-03-13" in remaining
            assert "lora_2026-03-14" in remaining
            # Only lora_2026-03-11 removed (not protected, not in top 3)
            assert result["removed"] == 1
            assert "lora_2026-03-11" not in remaining

    def test_fewer_than_max(self):
        """Fewer adapters than max_keep → nothing removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "lora"
            adapter_dir.mkdir()
            (adapter_dir / "lora_2026-03-14").mkdir()
            (adapter_dir / "lora_2026-03-15").mkdir()

            result = clean_lora_adapters(str(adapter_dir), max_keep=3)
            assert result["removed"] == 0
            assert result["retained"] == 2

    def test_empty_dir(self):
        """Empty directory → no errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "lora"
            adapter_dir.mkdir()

            result = clean_lora_adapters(str(adapter_dir))
            assert result["removed"] == 0
            assert result["retained"] == 0

    def test_no_dir(self):
        """Non-existent directory → no errors."""
        result = clean_lora_adapters("/nonexistent/path")
        assert result["removed"] == 0
        assert result["retained"] == 0
