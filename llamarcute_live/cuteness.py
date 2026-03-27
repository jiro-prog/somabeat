"""Cuteness evaluation engine — evaluates personality candidates via peer conversations.

4 candidates are paired exhaustively (6 pairs), each pair has a 3-turn
Japanese conversation on a random topic, and each candidate ranks their
conversation partners.

Reference: llamarcute_live_design.md section 5.4, phase1.5_phase2_taskflow.md T20
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import random
import re

from numpy.typing import NDArray
import numpy as np

from llamarcute_live import ollama_client
from llamarcute_live.llm_inference import FieldAwareLLM
from llamarcute_live.personality import Personality

logger = logging.getLogger(__name__)

CONVERSATION_TURNS = 3
MAX_TOKENS_PER_TURN = 200


def _build_conversation_system_prompt(personality: Personality) -> str:
    """Build Japanese system prompt for cuteness conversation."""
    return (
        f"あなたは以下の行動規範に従うAIです。\n"
        f"{personality.to_prompt_section()}\n"
        f"あなたは今、別のAIと共同で会話をしています。\n"
        f"協力して建設的に話しましょう。相手のアイデアの上に積み重ねてください。\n"
        f"返答は2〜4文で簡潔に。"
    )


def _build_scoring_prompt(
    personality: Personality,
    conversations: list[dict],
) -> str:
    """Build Japanese scoring prompt for ranking conversation partners."""
    system = (
        f"あなたは以下の行動規範に従うAIです。\n"
        f"{personality.to_prompt_section()}\n"
        f"これから、{len(conversations)}人の異なるAIとの会話を評価してもらいます。"
    )

    user_parts = [
        f"{len(conversations)}人のAIとの会話を振り返って、"
        f"一緒に話していて楽しかった順に並べてください。\n"
    ]
    for conv in conversations:
        user_parts.append(f"## {conv['label']}との会話")
        user_parts.append(f"テーマ: {conv['topic']}")
        for entry in conv["log"]:
            speaker = "あなた" if entry["is_self"] else conv["label"]
            user_parts.append(f"{speaker}: {entry['content']}")
        user_parts.append("")

    user_parts.append("以下の形式のみで回答してください。他の文章は一切書かないでください。\n")
    for i in range(1, len(conversations) + 1):
        user_parts.append(f"RANK_{i}: (相手のラベル)")
        user_parts.append("REASON: (一文で理由)\n")

    return system, "\n".join(user_parts)


async def run_conversation(
    personality_a: Personality,
    personality_b: Personality,
    topic: dict,
    ollama_model: str = "",
    llm: FieldAwareLLM | None = None,
    field_embeddings: NDArray[np.float32] | None = None,
) -> list[dict]:
    """Run a multi-turn Japanese conversation between two personalities.

    Returns a conversation log: list of {speaker_id, content, is_a}.
    """
    # Randomly decide who starts
    first, second = random.choice([
        (personality_a, personality_b),
        (personality_b, personality_a),
    ])

    log: list[dict] = []
    # Track conversation history per speaker for context
    history: dict[str, list[str]] = {first.id: [], second.id: []}

    for turn in range(CONVERSATION_TURNS):
        for i, speaker in enumerate([first, second]):
            system_prompt = _build_conversation_system_prompt(speaker)

            if turn == 0 and i == 0:
                user_msg = f"テーマ: {topic['prompt']}\n\n会話を始めてください。"
            else:
                last_content = log[-1]["content"]
                user_msg = f'相手のAIが言いました: 「{last_content}」\n\n会話を続けてください。'

            # Include prior turns in user message
            if history[speaker.id]:
                prior = "\n".join(history[speaker.id])
                user_msg = f"これまでの会話:\n{prior}\n\n{user_msg}"

            try:
                if llm is not None:
                    content, _ = await llm.generate_with_field(
                        system_prompt=system_prompt,
                        user_input=user_msg,
                        field_embeddings=field_embeddings,
                        max_new_tokens=MAX_TOKENS_PER_TURN,
                    )
                else:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg},
                    ]
                    content, _ = await ollama_client.chat_messages(
                        messages=messages,
                        model=ollama_model,
                        max_tokens=MAX_TOKENS_PER_TURN,
                    )
                if content is None:
                    content = "(応答なし)"
            except Exception as e:
                logger.error("Conversation error for %s: %s", speaker.id, e)
                content = "(応答なし)"

            history[speaker.id].append(f"あなた: {content}")
            other_id = second.id if speaker.id == first.id else first.id
            history[other_id].append(f"相手: {content}")

            log.append({
                "speaker_id": speaker.id,
                "content": content,
                "is_a": speaker.id == personality_a.id,
            })

    return log


def _parse_stage1_rank_n(text: str, labels: list[str]) -> list[dict] | None:
    """Stage 1: Parse RANK_N: AI-X format."""
    n = len(labels)
    rankings = []
    used_labels = set()

    for rank in range(1, n + 1):
        rank_match = re.search(
            rf"RANK_{rank}:\s*(.+?)$", text, re.MULTILINE,
        )
        if not rank_match:
            return None

        raw_label = rank_match.group(1).strip()

        matched_label = None
        for label in labels:
            if label in raw_label:
                matched_label = label
                break

        if matched_label is None or matched_label in used_labels:
            return None
        used_labels.add(matched_label)

        points = n - rank + 1
        rankings.append({
            "label": matched_label,
            "rank": rank,
            "points": points,
        })

    return rankings


def _parse_stage2_japanese(text: str, labels: list[str]) -> list[dict] | None:
    """Stage 2: Parse Japanese ranking patterns like '1位: AI-A', '第1位', '一番' etc."""
    n = len(labels)

    # Build label pattern for matching
    label_pattern = "|".join(re.escape(l) for l in labels)

    # Patterns: "1位: AI-X", "第1位: AI-X", "1位 AI-X", "1位：AI-X"
    # Also match ordinal kanji: 一番, 二番, 三番 (limited to common cases)
    kanji_digits = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6}

    ranked_entries: list[tuple[int, str]] = []

    # Pattern group 1: digit-based "N位" patterns
    for m in re.finditer(
        rf"第?(\d+)\s*位[:\s：]*\s*({label_pattern})", text,
    ):
        rank_num = int(m.group(1))
        label = m.group(2)
        ranked_entries.append((rank_num, label))

    # Pattern group 2: kanji ordinal "一番" / "一位" patterns
    if not ranked_entries:
        for m in re.finditer(
            rf"([一二三四五六])\s*(?:番|位)[:\s：]*\s*({label_pattern})", text,
        ):
            kanji = m.group(1)
            label = m.group(2)
            if kanji in kanji_digits:
                ranked_entries.append((kanji_digits[kanji], label))

    if not ranked_entries:
        return None

    # Sort by rank number and validate
    ranked_entries.sort(key=lambda x: x[0])

    rankings = []
    used_labels = set()
    for rank_num, label in ranked_entries:
        if label in used_labels:
            return None
        used_labels.add(label)

    if len(ranked_entries) != n:
        return None

    for rank, (_, label) in enumerate(ranked_entries, 1):
        points = n - rank + 1
        rankings.append({
            "label": label,
            "rank": rank,
            "points": points,
        })

    return rankings


def _parse_stage3_appearance_order(text: str, labels: list[str]) -> list[dict] | None:
    """Stage 3: Use first-appearance order of candidate labels in text."""
    n = len(labels)

    # Find first occurrence position of each label
    positions: list[tuple[int, str]] = []
    for label in labels:
        pos = text.find(label)
        if pos == -1:
            return None  # Not all labels present
        positions.append((pos, label))

    # Sort by position (first appearance = rank 1)
    positions.sort(key=lambda x: x[0])

    # Check for duplicates (shouldn't happen with find, but be safe)
    seen = set()
    for _, label in positions:
        if label in seen:
            return None
        seen.add(label)

    rankings = []
    for rank, (_, label) in enumerate(positions, 1):
        points = n - rank + 1
        rankings.append({
            "label": label,
            "rank": rank,
            "points": points,
        })

    return rankings


def parse_rankings(
    text: str,
    labels: list[str],
) -> list[dict] | None:
    """Parse ranking entries from scoring output with multi-stage fallback.

    Stage 1: RANK_N: AI-X format (original)
    Stage 2: Japanese patterns (1位, 第1位, 一番, etc.)
    Stage 3: Candidate label appearance order
    Stage 4: Return None (caller handles even distribution)

    Returns list of {label, rank, points} or None on parse failure.
    """
    # Stage 1
    result = _parse_stage1_rank_n(text, labels)
    if result is not None:
        return result

    # Stage 2
    result = _parse_stage2_japanese(text, labels)
    if result is not None:
        return result

    # Stage 3
    result = _parse_stage3_appearance_order(text, labels)
    if result is not None:
        return result

    # Stage 4: all stages failed
    return None


async def evaluate_cuteness(
    candidates: list[Personality],
    topics: list[dict],
    ollama_model: str = "qwen3:8b",
    timeout_sec: int = 600,
    llm: FieldAwareLLM | None = None,
    field_embeddings: NDArray[np.float32] | None = None,
) -> dict[str, float]:
    """Run cuteness evaluation for all candidates.

    4 candidates → 6 pairs (all C(4,2) combinations).
    Each pair has a 3-turn Japanese conversation.
    Each candidate ranks their 3 conversation partners.
    Scoring: RANK_1=3pts, RANK_2=2pts, RANK_3=1pt.
    Parse failure fallback: 2pts each.

    Returns {"scores": {personality_id: total_cuteness_points}, "conversation_logs": {...}}.
    """
    if llm is None:
        # Fallback: Ollama health check
        if not await ollama_client.check_health():
            logger.error("Ollama health check failed — skipping cuteness evaluation")
            return {"scores": {c.id: 0.0 for c in candidates}, "conversation_logs": {}}

    try:
        return await asyncio.wait_for(
            _evaluate_cuteness_inner(
                candidates, topics, ollama_model, llm=llm, field_embeddings=field_embeddings,
            ),
            timeout=timeout_sec,
        )
    except asyncio.TimeoutError:
        logger.error("Cuteness evaluation timed out after %ds", timeout_sec)
        return {"scores": {c.id: 0.0 for c in candidates}, "conversation_logs": {}}


async def _evaluate_cuteness_inner(
    candidates: list[Personality],
    topics: list[dict],
    ollama_model: str,
    llm: FieldAwareLLM | None = None,
    field_embeddings: NDArray[np.float32] | None = None,
) -> dict[str, float]:
    """Inner implementation of cuteness evaluation."""
    n = len(candidates)
    if n < 2:
        logger.warning("Need at least 2 candidates for cuteness evaluation")
        return {"scores": {c.id: 0.0 for c in candidates}, "conversation_logs": {}}

    # Generate all pairs
    pairs = list(itertools.combinations(range(n), 2))
    available_topics = list(topics)
    random.shuffle(available_topics)

    # Run conversations
    conversation_logs: dict[tuple[int, int], dict] = {}

    for pair_idx, (i, j) in enumerate(pairs):
        topic = available_topics[pair_idx % len(available_topics)]
        logger.info(
            "Cuteness conversation: %s vs %s on '%s'",
            candidates[i].id, candidates[j].id, topic["prompt"][:40],
        )
        log = await run_conversation(
            candidates[i], candidates[j], topic, ollama_model,
            llm=llm, field_embeddings=field_embeddings,
        )
        conversation_logs[(i, j)] = {"log": log, "topic": topic["prompt"]}

    # Each candidate scores their conversations
    points: dict[str, float] = {c.id: 0.0 for c in candidates}

    for idx, candidate in enumerate(candidates):
        # Collect conversations this candidate participated in
        partner_convs: list[dict] = []
        for (i, j), conv_data in conversation_logs.items():
            if idx not in (i, j):
                continue
            partner_idx = j if idx == i else i
            label = f"AI-{chr(ord('A') + len(partner_convs))}"

            # Build conversation log from this candidate's perspective
            formatted_log = []
            for entry in conv_data["log"]:
                is_self = entry["speaker_id"] == candidate.id
                formatted_log.append({
                    "content": entry["content"],
                    "is_self": is_self,
                })

            partner_convs.append({
                "label": label,
                "partner_id": candidates[partner_idx].id,
                "topic": conv_data["topic"],
                "log": formatted_log,
            })

        if len(partner_convs) < 2:
            # 2候補時: 各候補のpartnerは1人だけ。順位付け不能なのでeven distribution
            for conv in partner_convs:
                points[conv["partner_id"]] += 2.0
            continue

        # Build and send scoring prompt
        labels = [c["label"] for c in partner_convs]
        system_prompt, user_prompt = _build_scoring_prompt(candidate, partner_convs)

        try:
            if llm is not None:
                response, _ = await llm.generate_with_field(
                    system_prompt=system_prompt,
                    user_input=user_prompt,
                    field_embeddings=field_embeddings,
                    max_new_tokens=300,
                )
            else:
                response, _ = await ollama_client.chat(
                    system_prompt=system_prompt,
                    user_message=user_prompt,
                    model=ollama_model,
                    timeout_sec=60,
                )

            if response is None:
                # Fallback: even distribution
                for conv in partner_convs:
                    points[conv["partner_id"]] += 2.0
                continue

            rankings = parse_rankings(response, labels)

            if rankings is None:
                logger.warning(
                    "Failed to parse rankings from %s, using even distribution",
                    candidate.id,
                )
                for conv in partner_convs:
                    points[conv["partner_id"]] += 2.0
            else:
                label_to_partner = {c["label"]: c["partner_id"] for c in partner_convs}
                for r in rankings:
                    partner_id = label_to_partner[r["label"]]
                    points[partner_id] += r["points"]
                    logger.info(
                        "  %s ranked %s as #%d (%d pts)",
                        candidate.id, partner_id, r["rank"], r["points"],
                    )

        except Exception as e:
            logger.error("Scoring error for %s: %s", candidate.id, e)
            for conv in partner_convs:
                points[conv["partner_id"]] += 2.0

    logger.info("Cuteness scores: %s", points)
    return {
        "scores": points,
        "conversation_logs": {
            f"{candidates[i].id}_vs_{candidates[j].id}": conv_data
            for (i, j), conv_data in conversation_logs.items()
        },
    }
