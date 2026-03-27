"""Tests for bridge module (sleep_ingest / wake_export)."""

import asyncio
import json
import tempfile
from pathlib import Path

import aiosqlite
import numpy as np
import pytest

from shared_state.backends.chromadb_backend import ChromaDBField
from shared_state.interface import PerceiveParams, Signal, SignalOrigin
from shared_state.observer import LoggingObserver


class MockEncoder:
    """Mock encoder returning a fixed unit vector so all signals have similarity ~1.0."""
    dimensionality = 384
    def _fixed(self):
        v = np.ones(384, dtype=np.float32)
        return v / np.linalg.norm(v)
    def encode(self, text): return self._fixed()
    def encode_for_emit(self, text): return self._fixed()
    def encode_for_sense(self, text): return self._fixed()


def _make_field():
    import uuid
    return ChromaDBField(
        persist_directory=None,
        collection_name=f"test_{uuid.uuid4().hex[:8]}",
        observer=LoggingObserver(),
    )


async def _setup_llamarcute_db(db_path):
    async with aiosqlite.connect(db_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS dialogue_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_dialogue_created ON dialogue_log(created_at)")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS emit_log (
                signal_id TEXT PRIMARY KEY,
                context TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS rotation_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                instruction TEXT NOT NULL,
                output TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
            )
        """)
        await db.execute("INSERT INTO dialogue_log (role, content) VALUES ('user', 'こんにちは')")
        await db.execute("INSERT INTO dialogue_log (role, content) VALUES ('assistant', 'やあ！')")
        await db.execute("INSERT INTO dialogue_log (role, content) VALUES ('user', '量子力学って何？')")
        await db.execute("INSERT INTO dialogue_log (role, content) VALUES ('assistant', 'うーん、難しいね')")
        await db.commit()


async def _setup_sleepyjean_db(db_path):
    async with aiosqlite.connect(db_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS homework (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                theme TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                requested_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                completed_at TEXT,
                result_summary TEXT,
                retry_count INTEGER NOT NULL DEFAULT 0
            )
        """)
        await db.commit()


class TestSleepIngest:
    def test_transfer_dialogue(self):
        from bridge.sleep_ingest import sleep_ingest

        async def run():
            with tempfile.TemporaryDirectory() as tmpdir:
                ll_db = Path(tmpdir) / "ll.db"
                sj_db = Path(tmpdir) / "sj.db"
                ki = Path(tmpdir) / "ki.json"
                ki_bak = Path(tmpdir) / "ki_before.json"

                await _setup_llamarcute_db(ll_db)
                await _setup_sleepyjean_db(sj_db)
                ki.write_text(json.dumps({"topics": [], "open_questions": []}))

                field = _make_field()

                result = await sleep_ingest(
                    field=field,
                    encoder=MockEncoder(),
                    llamarcute_db_path=ll_db,
                    sleepyjean_db_path=sj_db,
                    knowledge_index_path=ki,
                    knowledge_index_backup_path=ki_bak,
                )

                assert result["dialogue_transferred"] == 4
                assert result["knowledge_snapshot_saved"] is True
                assert ki_bak.exists()

                async with aiosqlite.connect(sj_db) as db:
                    cursor = await db.execute("SELECT COUNT(*) FROM conversations")
                    count = (await cursor.fetchone())[0]
                    assert count == 4

        asyncio.get_event_loop().run_until_complete(run())


    def test_difficulty_dedup_skips_existing_homework(self):
        """Duplicate homework themes (queued or learned) should not be re-inserted."""
        from bridge.sleep_ingest import sleep_ingest

        async def run():
            with tempfile.TemporaryDirectory() as tmpdir:
                ll_db = Path(tmpdir) / "ll.db"
                sj_db = Path(tmpdir) / "sj.db"
                ki = Path(tmpdir) / "ki.json"
                ki_bak = Path(tmpdir) / "ki_before.json"

                await _setup_llamarcute_db(ll_db)
                await _setup_sleepyjean_db(sj_db)
                ki.write_text(json.dumps({"topics": [], "open_questions": []}))

                field = _make_field()
                encoder = MockEncoder()

                # Emit a difficulty signal
                sig = Signal.create(
                    embedding=encoder.encode_for_emit("difficult topic"),
                    origin=SignalOrigin(system="llamarcute_live", context="difficulty"),
                )
                await field.emit(sig)

                # First ingest — should add 1 homework
                result1 = await sleep_ingest(
                    field=field, encoder=encoder,
                    llamarcute_db_path=ll_db, sleepyjean_db_path=sj_db,
                    knowledge_index_path=ki, knowledge_index_backup_path=ki_bak,
                    difficulty_threshold=0.0,
                )
                assert result1["homework_added"] == 1

                # Second ingest — same signal, should be skipped (duplicate)
                result2 = await sleep_ingest(
                    field=field, encoder=encoder,
                    llamarcute_db_path=ll_db, sleepyjean_db_path=sj_db,
                    knowledge_index_path=ki, knowledge_index_backup_path=ki_bak,
                    difficulty_threshold=0.0,
                )
                assert result2["homework_added"] == 0

                # Verify only 1 homework entry exists
                async with aiosqlite.connect(sj_db) as db:
                    cursor = await db.execute("SELECT COUNT(*) FROM homework")
                    count = (await cursor.fetchone())[0]
                    assert count == 1

        asyncio.get_event_loop().run_until_complete(run())

    def test_difficulty_time_horizon_filters_old_signals(self):
        """Only difficulty signals within 25h should be ingested."""
        from datetime import datetime, timedelta
        from bridge.sleep_ingest import sleep_ingest

        async def run():
            with tempfile.TemporaryDirectory() as tmpdir:
                ll_db = Path(tmpdir) / "ll.db"
                sj_db = Path(tmpdir) / "sj.db"
                ki = Path(tmpdir) / "ki.json"
                ki_bak = Path(tmpdir) / "ki_before.json"

                await _setup_llamarcute_db(ll_db)
                await _setup_sleepyjean_db(sj_db)
                ki.write_text(json.dumps({"topics": [], "open_questions": []}))

                field = _make_field()
                encoder = MockEncoder()

                # Emit an old difficulty signal (2 days ago)
                old_sig = Signal.create(
                    embedding=encoder.encode_for_emit("old difficulty"),
                    origin=SignalOrigin(system="llamarcute_live", context="difficulty"),
                )
                await field.emit(old_sig)

                # Backdate the signal in ChromaDB
                old_time = (datetime.now() - timedelta(days=2)).isoformat()
                field._collection.update(
                    ids=[old_sig.signal_id],
                    metadatas=[{
                        "origin_system": "llamarcute_live",
                        "origin_context": "difficulty",
                        "emitted_at": old_time,
                        "norm": 1.0,
                    }],
                )

                # Emit a recent difficulty signal
                new_sig = Signal.create(
                    embedding=encoder.encode_for_emit("new difficulty"),
                    origin=SignalOrigin(system="llamarcute_live", context="difficulty"),
                )
                await field.emit(new_sig)

                result = await sleep_ingest(
                    field=field, encoder=encoder,
                    llamarcute_db_path=ll_db, sleepyjean_db_path=sj_db,
                    knowledge_index_path=ki, knowledge_index_backup_path=ki_bak,
                    difficulty_threshold=0.0,
                )

                # Only the recent signal should be ingested
                assert result["homework_added"] == 1

                async with aiosqlite.connect(sj_db) as db:
                    cursor = await db.execute("SELECT theme FROM homework")
                    rows = await cursor.fetchall()
                    assert len(rows) == 1
                    # T1: theme is now signal_id-based (difficulty:<id>)
                    assert rows[0][0].startswith("difficulty:")

        asyncio.get_event_loop().run_until_complete(run())


    def test_difficulty_not_crowded_out_by_knowledge_update(self):
        """Difficulty signals must be found even when knowledge_update signals
        have higher cosine similarity to the query (the bug this fix addresses)."""
        from bridge.sleep_ingest import sleep_ingest

        async def run():
            with tempfile.TemporaryDirectory() as tmpdir:
                ll_db = Path(tmpdir) / "ll.db"
                sj_db = Path(tmpdir) / "sj.db"
                ki = Path(tmpdir) / "ki.json"
                ki_bak = Path(tmpdir) / "ki_before.json"

                await _setup_llamarcute_db(ll_db)
                await _setup_sleepyjean_db(sj_db)
                ki.write_text(json.dumps({"topics": [], "open_questions": []}))

                field = _make_field()
                encoder = MockEncoder()

                # Emit many knowledge_update signals (norm=2.0, like wake_export)
                for i in range(15):
                    emb = encoder.encode_for_emit(f"learned topic {i}")
                    emb = emb * 2.0  # norm 2.0, same as wake_export
                    sig = Signal.create(
                        embedding=emb,
                        origin=SignalOrigin(system="sleepyjean", context="knowledge_update"),
                    )
                    await field.emit(sig)

                # Emit a few difficulty signals
                for i in range(3):
                    sig = Signal.create(
                        embedding=encoder.encode_for_emit(f"difficulty {i}"),
                        origin=SignalOrigin(system="llamarcute_live", context="difficulty"),
                    )
                    await field.emit(sig)

                result = await sleep_ingest(
                    field=field, encoder=encoder,
                    llamarcute_db_path=ll_db, sleepyjean_db_path=sj_db,
                    knowledge_index_path=ki, knowledge_index_backup_path=ki_bak,
                )

                # All 3 difficulty signals should be picked up
                assert result["homework_added"] == 3

        asyncio.get_event_loop().run_until_complete(run())


class TestWakeExport:
    def test_new_topic_emitted(self):
        from bridge.wake_export import wake_export

        async def run():
            with tempfile.TemporaryDirectory() as tmpdir:
                ki = Path(tmpdir) / "ki.json"
                ki_bak = Path(tmpdir) / "ki_before.json"

                before = {"topics": [], "open_questions": []}
                after = {
                    "topics": [{"id": "topic_001", "name": "量子力学", "confidence": 0.65}],
                    "open_questions": [],
                }

                ki_bak.write_text(json.dumps(before))
                ki.write_text(json.dumps(after))

                field = _make_field()
                result = await wake_export(
                    field=field, encoder=MockEncoder(),
                    knowledge_index_path=ki, knowledge_index_backup_path=ki_bak,
                )
                assert result["new_topics_emitted"] == 1

                snap = await field.snapshot()
                assert snap.total_count >= 1

        asyncio.get_event_loop().run_until_complete(run())

    def test_confidence_change_emitted(self):
        from bridge.wake_export import wake_export

        async def run():
            with tempfile.TemporaryDirectory() as tmpdir:
                ki = Path(tmpdir) / "ki.json"
                ki_bak = Path(tmpdir) / "ki_before.json"

                before = {"topics": [{"id": "t1", "name": "Python", "confidence": 0.4}], "open_questions": []}
                after = {"topics": [{"id": "t1", "name": "Python", "confidence": 0.7}], "open_questions": []}

                ki_bak.write_text(json.dumps(before))
                ki.write_text(json.dumps(after))

                field = _make_field()
                result = await wake_export(
                    field=field, encoder=MockEncoder(),
                    knowledge_index_path=ki, knowledge_index_backup_path=ki_bak,
                )
                assert result["confidence_changes_emitted"] == 1

        asyncio.get_event_loop().run_until_complete(run())

    def test_deleted_topic_emitted(self):
        from bridge.wake_export import wake_export

        async def run():
            with tempfile.TemporaryDirectory() as tmpdir:
                ki = Path(tmpdir) / "ki.json"
                ki_bak = Path(tmpdir) / "ki_before.json"

                before = {"topics": [{"id": "t1", "name": "忘れるトピック", "confidence": 0.3}], "open_questions": []}
                after = {"topics": [], "open_questions": []}

                ki_bak.write_text(json.dumps(before))
                ki.write_text(json.dumps(after))

                field = _make_field()
                result = await wake_export(
                    field=field, encoder=MockEncoder(),
                    knowledge_index_path=ki, knowledge_index_backup_path=ki_bak,
                )
                assert result["deleted_topics_emitted"] == 1

        asyncio.get_event_loop().run_until_complete(run())


class TestSelectRotationQA:
    """Tests for Q&A rotation selection (T13)."""

    def test_limits_count(self):
        from bridge.wake_export import select_rotation_qa

        # 20 Q&A across 4 files
        pairs = []
        for i in range(5):
            for fname in ["sft_topic_a", "sft_topic_b", "sft_topic_c", "sft_topic_d"]:
                pairs.append({
                    "instruction": f"Question {i} from {fname}",
                    "output": f"Answer {i}",
                    "source_file": fname,
                })

        selected = select_rotation_qa(pairs, max_count=8)
        assert len(selected) == 8

        # All 4 files should be represented
        files_represented = {qa["source_file"] for qa in selected}
        assert len(files_represented) == 4

    def test_fewer_than_max_returns_all(self):
        from bridge.wake_export import select_rotation_qa

        pairs = [
            {"instruction": "Q1", "output": "A1", "source_file": "sft_a"},
            {"instruction": "Q2", "output": "A2", "source_file": "sft_b"},
            {"instruction": "Q3", "output": "A3", "source_file": "sft_b"},
        ]

        selected = select_rotation_qa(pairs, max_count=8)
        assert len(selected) == 3

    def test_single_file_limits_correctly(self):
        from bridge.wake_export import select_rotation_qa

        # All from one file
        pairs = [
            {"instruction": f"Q{i}", "output": f"A{i}", "source_file": "sft_only"}
            for i in range(15)
        ]

        selected = select_rotation_qa(pairs, max_count=8)
        assert len(selected) == 8

    def test_emit_qa_pairs_respects_limit(self):
        """Integration test: _emit_qa_pairs respects max_qa limit."""
        from bridge.wake_export import _emit_qa_pairs

        async def run():
            with tempfile.TemporaryDirectory() as tmpdir:
                # Create fake JSONL with many entries
                from datetime import date
                today_dir = Path(tmpdir) / date.today().isoformat()
                today_dir.mkdir()

                for topic in ["physics", "biology"]:
                    fpath = today_dir / f"sft_{topic}.jsonl"
                    with open(fpath, "w") as f:
                        for i in range(12):
                            record = {"instruction": f"{topic} question {i}", "output": f"answer {i}"}
                            f.write(json.dumps(record) + "\n")

                field = _make_field()
                count = await _emit_qa_pairs(field, MockEncoder(), tmpdir, max_qa=8)
                assert count <= 8

                snap = await field.snapshot()
                assert snap.total_count <= 8

        asyncio.get_event_loop().run_until_complete(run())

    def test_emit_qa_pairs_stores_output_in_extra(self):
        """Q&A signals should store instruction and output in signal.extra."""
        from bridge.wake_export import _emit_qa_pairs

        async def run():
            with tempfile.TemporaryDirectory() as tmpdir:
                from datetime import date
                today_dir = Path(tmpdir) / date.today().isoformat()
                today_dir.mkdir()

                fpath = today_dir / "sft_test.jsonl"
                with open(fpath, "w") as f:
                    record = {"instruction": "What is Python?", "output": "A programming language"}
                    f.write(json.dumps(record) + "\n")

                field = _make_field()
                encoder = MockEncoder()
                count = await _emit_qa_pairs(field, encoder, tmpdir, max_qa=8)
                assert count == 1

                # Perceive and verify extra contains output
                perception = await field.perceive(PerceiveParams(max_signals=10, min_strength=0.0))
                assert len(perception.signals) == 1

                sig = perception.signals[0].signal
                assert sig.extra.get("instruction") == "What is Python?"
                assert sig.extra.get("output") == "A programming language"

        asyncio.get_event_loop().run_until_complete(run())


class TestPurgePolicy:
    """Tests for purge in orchestrator (T14)."""

    def test_purge_removes_old_signals(self):
        from datetime import datetime, timedelta
        from shared_state.interface import PurgeCriteria

        async def run():
            field = _make_field()
            encoder = MockEncoder()

            # Emit an old signal (fake the timestamp)
            old_signal = Signal.create(
                embedding=encoder.encode("old data"),
                origin=SignalOrigin(system="test", context="old"),
            )
            await field.emit(old_signal)

            # Manually update the metadata to make it old
            field._collection.update(
                ids=[old_signal.signal_id],
                metadatas=[{
                    "origin_system": "test",
                    "origin_context": "old",
                    "emitted_at": (datetime.now() - timedelta(days=10)).isoformat(),
                    "norm": 1.0,
                }],
            )

            # Emit a new signal
            new_signal = Signal.create(
                embedding=encoder.encode("new data"),
                origin=SignalOrigin(system="test", context="new"),
            )
            await field.emit(new_signal)

            # Purge with 7 day max age
            criteria = PurgeCriteria(older_than=timedelta(days=7))
            result = await field.purge(criteria)

            assert result.purged_count == 1
            assert result.remaining_count == 1

        asyncio.get_event_loop().run_until_complete(run())
