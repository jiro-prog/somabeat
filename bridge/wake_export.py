"""Wake export bridge — transfers SleepyJean's learning results to the shared field.

Runs after SleepyJean's night cycle, before the wake-up sequence.

Reference: llamarcute_live_design.md section 6.4, phase1_taskflow.md T7
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from shared_state.backends.chromadb_backend import ChromaDBField
from shared_state.encoder import E5SmallEncoder
from shared_state.interface import Signal, SignalOrigin

logger = logging.getLogger(__name__)

SLEEPYJEAN_ORIGIN = SignalOrigin(system="sleepyjean", context="knowledge_update")
QA_ORIGIN = SignalOrigin(system="sleepyjean", context="rotation_task")
DREAM_ORIGIN = SignalOrigin(system="sleepyjean", context="dream")


async def wake_export(
    field: ChromaDBField,
    encoder: E5SmallEncoder,
    knowledge_index_path: str | Path,
    knowledge_index_backup_path: str | Path,
    night_result_dir: str | Path | None = None,
) -> dict:
    """Export SleepyJean's learning results to the shared field.

    Returns a summary dict for logging.
    """
    summary = {
        "new_topics_emitted": 0,
        "confidence_changes_emitted": 0,
        "deleted_topics_emitted": 0,
        "qa_pairs_emitted": 0,
        "dream_signals_emitted": 0,
    }

    # 1. Compare knowledge_index before/after
    try:
        ki_path = Path(knowledge_index_path)
        backup_path = Path(knowledge_index_backup_path)

        if not ki_path.exists():
            logger.warning("wake_export: knowledge_index.json not found")
            return summary

        after = _load_json(ki_path)

        if backup_path.exists():
            before = _load_json(backup_path)
        else:
            before = {"topics": [], "open_questions": []}
            logger.warning("wake_export: No backup snapshot, treating as empty")

        # Diff topics
        before_topics = {t["id"]: t for t in before.get("topics", [])}
        after_topics = {t["id"]: t for t in after.get("topics", [])}

        # New topics
        for tid, topic in after_topics.items():
            if tid not in before_topics:
                trace = f"トピック「{topic['name']}」を新たに学習した"
                embedding = encoder.encode_for_emit(trace)
                embedding = embedding * 2.0  # norm 2.0 for new topics
                signal = Signal.create(embedding=embedding, origin=SLEEPYJEAN_ORIGIN, trace=trace)
                await field.emit(signal)
                summary["new_topics_emitted"] += 1

        # Confidence changes
        for tid, topic in after_topics.items():
            if tid in before_topics:
                old_conf = before_topics[tid].get("confidence", 0)
                new_conf = topic.get("confidence", 0)
                delta = new_conf - old_conf
                if abs(delta) > 0.05:
                    trace = f"トピック「{topic['name']}」の確信度が {old_conf:.2f}→{new_conf:.2f} に変化"
                    embedding = encoder.encode_for_emit(trace)
                    norm_scale = 1.0 + abs(delta) * 5  # proportional to change
                    embedding = embedding * norm_scale
                    signal = Signal.create(embedding=embedding, origin=SLEEPYJEAN_ORIGIN, trace=trace)
                    await field.emit(signal)
                    summary["confidence_changes_emitted"] += 1

        # Deleted topics
        for tid, topic in before_topics.items():
            if tid not in after_topics:
                trace = f"トピック「{topic['name']}」の記憶を忘却した"
                embedding = encoder.encode_for_emit(trace)
                signal = Signal.create(embedding=embedding, origin=SLEEPYJEAN_ORIGIN, trace=trace)
                await field.emit(signal)
                summary["deleted_topics_emitted"] += 1

        # New open questions
        before_qs = {q["question"] for q in before.get("open_questions", [])}
        for q in after.get("open_questions", []):
            if q["question"] not in before_qs:
                trace = f"新たな疑問: {q['question']}"
                embedding = encoder.encode_for_emit(trace)
                signal = Signal.create(embedding=embedding, origin=SLEEPYJEAN_ORIGIN, trace=trace)
                await field.emit(signal)

        logger.info(
            "wake_export: topics new=%d conf_change=%d deleted=%d",
            summary["new_topics_emitted"],
            summary["confidence_changes_emitted"],
            summary["deleted_topics_emitted"],
        )

    except Exception as e:
        logger.error("wake_export: Knowledge diff failed: %s", e)

    # 2. Emit Q&A pairs as rotation tasks (from night result log)
    if night_result_dir:
        try:
            qa_count = await _emit_qa_pairs(field, encoder, night_result_dir)
            summary["qa_pairs_emitted"] = qa_count
            logger.info("wake_export: Emitted %d Q&A rotation tasks", qa_count)
        except Exception as e:
            logger.error("wake_export: Q&A emission failed: %s", e)

    # 3. Emit dream insights as signals (T3: dream tracking)
    if night_result_dir:
        try:
            dream_count = await _emit_dream_signals(field, encoder, night_result_dir)
            summary["dream_signals_emitted"] = dream_count
            logger.info("wake_export: Emitted %d dream signals", dream_count)
        except Exception as e:
            logger.error("wake_export: Dream emission failed: %s", e)

    return summary


def select_rotation_qa(
    qa_pairs: list[dict],
    max_count: int = 8,
) -> list[dict]:
    """Select diverse Q&A pairs by round-robin sampling from source files.

    Groups qa_pairs by source_file, then round-robin picks one per file
    until max_count is reached. This ensures topic diversity since each
    SleepyJean JSONL file corresponds to a different topic.

    Args:
        qa_pairs: List of dicts with keys: instruction, output, source_file.
        max_count: Maximum number of Q&A pairs to return.

    Returns:
        Selected subset of qa_pairs, at most max_count items.
    """
    if len(qa_pairs) <= max_count:
        return qa_pairs

    # Group by source file
    groups: dict[str, list[dict]] = defaultdict(list)
    for qa in qa_pairs:
        groups[qa["source_file"]].append(qa)

    # Shuffle within each group for variety
    for file_pairs in groups.values():
        random.shuffle(file_pairs)

    # Round-robin across groups
    selected: list[dict] = []
    group_keys = list(groups.keys())
    random.shuffle(group_keys)  # randomize group order too
    group_iterators = {k: iter(groups[k]) for k in group_keys}

    while len(selected) < max_count:
        exhausted = []
        for key in group_keys:
            if len(selected) >= max_count:
                break
            try:
                selected.append(next(group_iterators[key]))
            except StopIteration:
                exhausted.append(key)
        # Remove exhausted groups
        for key in exhausted:
            group_keys.remove(key)
        if not group_keys:
            break

    return selected


async def _emit_qa_pairs(
    field: ChromaDBField,
    encoder: E5SmallEncoder,
    night_result_dir: str | Path,
    max_qa: int = 8,
) -> int:
    """Emit quality-checked Q&A pairs from SleepyJean's training data.

    Collects all valid Q&A pairs, selects a diverse subset (up to max_qa),
    and emits only those to the shared field.
    """
    from datetime import date
    import glob

    today = date.today().isoformat()
    pattern = str(Path(night_result_dir) / today / "sft_*.jsonl")
    files = glob.glob(pattern)

    # 1. Collect all valid Q&A pairs
    all_pairs: list[dict] = []
    for fpath in files:
        source_file = Path(fpath).stem  # e.g. "sft_quantum_mechanics"
        with open(fpath) as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                instruction = record.get("instruction", "")
                output = record.get("output", "")
                if not instruction or not output:
                    continue

                all_pairs.append({
                    "instruction": instruction,
                    "output": output,
                    "source_file": source_file,
                })

    logger.info(
        "_emit_qa_pairs: Found %d Q&A pairs across %d files, selecting up to %d",
        len(all_pairs), len(files), max_qa,
    )

    # 2. Select diverse subset
    selected = select_rotation_qa(all_pairs, max_count=max_qa)

    # 3. Emit selected pairs (instruction in trace, output in extra)
    count = 0
    for qa in selected:
        trace = f"Q&A: {qa['instruction'][:80]}"
        embedding = encoder.encode_for_emit(qa["instruction"])
        signal = Signal.create(
            embedding=embedding, origin=QA_ORIGIN, trace=trace,
            extra={"instruction": qa["instruction"], "output": qa.get("output", "")},
        )
        await field.emit(signal)
        count += 1

    return count


async def _emit_dream_signals(
    field: ChromaDBField,
    encoder: E5SmallEncoder,
    night_result_dir: str | Path,
) -> int:
    """Emit dream diary insights as signals to the shared field.

    Reads the latest night result JSON and emits dream connections/insights
    with context="dream" so metrics can track dream influence on dialogue.

    Note: night_result_dir is the training data dir (data/training/).
    The night result JSON is in logs/growth/ under the SleepyJean root.
    We derive the SleepyJean root from the training dir path:
      .../sleepyjean/data/training → .../sleepyjean
    """
    from datetime import date

    today = date.today().isoformat()
    # Derive SleepyJean root explicitly: training dir is always <sj_root>/data/training
    sj_root = Path(night_result_dir).parent.parent
    logs_growth_dir = sj_root / "logs" / "growth"
    result_path = logs_growth_dir / f"night_{today}.json"

    if not result_path.exists():
        logger.debug("_emit_dream_signals: No night result for %s", today)
        return 0

    night_result = _load_json(result_path)
    dream = night_result.get("dream", {})
    if not dream:
        return 0

    count = 0

    # Emit dream connections (topic cross-links)
    for conn in dream.get("connections", []):
        topics = conn.get("topics", [])
        insight = conn.get("insight", "")
        if not insight:
            continue
        trace = f"夢の接続: {' × '.join(topics[:3])} — {insight[:80]}"
        embedding = encoder.encode_for_emit(trace)
        signal = Signal.create(embedding=embedding, origin=DREAM_ORIGIN, trace=trace)
        await field.emit(signal)
        count += 1

    # Emit dream insights
    for insight in dream.get("insights", []):
        trace = f"夢の洞察: {insight[:100]}"
        embedding = encoder.encode_for_emit(trace)
        signal = Signal.create(embedding=embedding, origin=DREAM_ORIGIN, trace=trace)
        await field.emit(signal)
        count += 1

    return count


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)
