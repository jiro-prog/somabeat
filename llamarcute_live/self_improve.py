"""Self-improvement engine — generates mutation candidates for personality evolution.

Adapted from llamarcute/src/self_improve.py for the integrated system.
The mutation process runs during sleep, after SleepyJean's night cycle.

Reference: llamarcute_live_design.md section 5.4, phase1.5_phase2_taskflow.md T18
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import timedelta

from shared_state.backends.chromadb_backend import ChromaDBField
from shared_state.encoder import E5SmallEncoder
from shared_state.field_receptor import FieldReceptorImpl
from shared_state.interface import PerceiveParams, SenseParams, SignalOrigin

from llamarcute_live.llm_inference import FieldAwareLLM
from llamarcute_live.personality import Personality

logger = logging.getLogger(__name__)

SELF_IMPROVE_ORIGIN = SignalOrigin(system="llamarcute_live", context="self_improvement")

DIVERSITY_INSTRUCTIONS = [
    "Focus on fixing weaknesses. Analyze the difficulties and propose direct countermeasures.",
    "Try a new approach. Introduce a different methodology from the current framework.",
    "Focus on rebalancing. Adjust the priorities and wording of existing rules.",
]

MUTATION_SYSTEM_PROMPT = """\
You are an AI strategy optimizer. Analyze performance results \
and propose improvements to the strategy rules.

Rules are identified by number (R1, R2, ...). Use these IDs to reference rules.

You must respond in this exact format:

ANALYSIS:
(Your analysis of what went wrong and what can be improved)

CHANGES:
- action: modify
  rule_id: R3
  new: "(new rule text)"
  reason: "(why)"
- action: delete
  rule_id: R7
  reason: "(why)"
- action: add
  category: reasoning|response|meta|tone|knowledge_attitude
  new: "(new rule text)"
  reason: "(why)"
"""

MUTATION_USER_TEMPLATE = """\
## Out of Scope
The following are immutable constraints and must NOT be mutated, deleted, or modified:
- Response language (Japanese)
- Response length limit (Discord 2000 characters)
Only behavioral_rules are subject to change.

## Current Strategy Rules
{numbered_rules}

## Recent Difficulties (signals from the shared field)
{difficulties}

## Recent Learning (from SleepyJean's knowledge updates)
{memories}

## Constraints
- You may change up to {max_changes} rules (add, modify, or delete)
- Total rules must remain at least 5
- Total token count of all rules must stay under 800 tokens
- Current rule count: {rule_count}. Consider modifying or consolidating existing rules rather than only adding new ones.
- If total rules exceed 12, prioritize deletion or merging of redundant rules.
- For modify/delete, reference rules by their ID (e.g., R1, R2). Do NOT copy the full rule text.

## Direction
{diversity_instruction}

Propose specific rule changes. For each change, explain your reasoning.
"""


def parse_changes(text: str) -> list[dict] | None:
    """Parse CHANGES section from self-improvement output.

    Supports rule_id references (R1, R2, ...) for modify/delete actions.
    For add actions, category is required instead of rule_id.
    """
    changes_match = re.search(r"CHANGES:\s*\n(.*)", text, re.DOTALL)
    if not changes_match:
        return None

    changes_text = changes_match.group(1)
    changes = []
    entries = re.split(r"(?=^- action:)", changes_text, flags=re.MULTILINE)

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        change: dict[str, str] = {}
        for field_name in ["action", "category", "rule_id", "original", "new", "reason"]:
            match = re.search(
                rf'^[\s\-]*{field_name}:\s*"?(.*?)"?\s*$', entry, re.MULTILINE,
            )
            if match:
                val = match.group(1).strip().strip('"')
                if val:
                    change[field_name] = val
        if "action" not in change:
            continue
        action = change["action"].lower()
        # modify/delete need rule_id (or legacy original+category)
        # add needs category
        if action in ("modify", "delete") and ("rule_id" in change or "category" in change):
            changes.append(change)
        elif action == "add" and "category" in change:
            changes.append(change)

    return changes if changes else None


@dataclass
class MaxChangesResult:
    """Result of determine_max_changes with fallback tracking."""
    max_changes: int
    conservative_mode_active: bool = False
    difficulty_based_value: int = 1  # value before immune override


async def determine_max_changes(
    field: ChromaDBField,
    encoder: E5SmallEncoder,
    difficulty_time_horizon_hours: int = 24,
) -> MaxChangesResult:
    """Determine max rule changes based on difficulty signal accumulation.

    More accumulated difficulty signals → more aggressive changes.
    - 0-2 difficulty signals with weight > 0.3: conservative (1 change)
    - 3-5: moderate (2 changes)
    - 6+: aggressive (3 changes)

    If a conservative_mode signal from the immune system is present,
    always returns 1 regardless of difficulty count.

    Returns MaxChangesResult with fallback tracking info.

    Reference: llamarcute_live_design.md section 5.3
    """
    conservative_mode_active = False

    # Check for immune system conservative_mode signal
    # Uses snapshot() + context filter to avoid cosine-similarity crowding
    from datetime import datetime
    now = datetime.now()

    snap = await field.snapshot()

    for s in snap.signals:
        if (
            s.origin.context == "immune:conservative_mode"
            and (now - s.emitted_at) < timedelta(hours=48)
        ):
            logger.info(
                "Conservative mode active (immune signal), forcing max_changes=1",
            )
            conservative_mode_active = True
            break

    # Count difficulty signals using snapshot() + context filter
    # (same pattern as rotation_score and sleep_ingest fixes)
    horizon = timedelta(hours=difficulty_time_horizon_hours)
    difficulty_count = sum(
        1 for s in snap.signals
        if s.origin.context == "difficulty"
        and (now - s.emitted_at) < horizon
    )

    logger.info(
        "Difficulty signals (last %dh): %d",
        difficulty_time_horizon_hours, difficulty_count,
    )

    if difficulty_count >= 6:
        difficulty_based = 3
    elif difficulty_count >= 3:
        difficulty_based = 2
    else:
        difficulty_based = 1

    final = 1 if conservative_mode_active else difficulty_based

    return MaxChangesResult(
        max_changes=final,
        conservative_mode_active=conservative_mode_active,
        difficulty_based_value=difficulty_based,
    )


async def generate_mutation_candidates(
    current_personality: Personality,
    max_changes: int,
    field: ChromaDBField,
    encoder: E5SmallEncoder,
    llm: FieldAwareLLM | None = None,
    receptor: FieldReceptorImpl | None = None,
    num_candidates: int = 3,
) -> list[Personality]:
    """Generate mutation candidates using field perception as context.

    Steps:
    1. Perceive field → FieldReceptor → embedding injection
    2. For each of 3 diversity instructions, generate a mutation via LLM
    3. Parse and validate each mutation
    4. Return list of valid mutated Personalities

    Returns at most num_candidates Personalities. May return fewer if
    some mutations fail to parse or validate.
    """
    # 1. Perceive field and transform via FieldReceptor
    field_embeddings = None
    if receptor is not None:
        perception = await field.perceive(PerceiveParams())
        if perception.signals:
            field_embeddings = receptor.transduce(
                [ps.signal.embedding for ps in perception.signals],
                [ps.strength for ps in perception.signals],
            )

    current_ver = current_personality.version
    next_ver = current_ver + 1

    # 2. Generate candidates
    candidates: list[Personality] = []

    for i, instruction in enumerate(DIVERSITY_INSTRUCTIONS[:num_candidates]):
        candidate_id = f"mutation_{next_ver}_{chr(ord('A') + i)}"

        user_prompt = MUTATION_USER_TEMPLATE.format(
            numbered_rules=current_personality.to_numbered_rules(),
            difficulties="(perceived via field embedding injection)",
            memories="(perceived via field embedding injection)",
            max_changes=max_changes,
            rule_count=current_personality.rule_count,
            diversity_instruction=instruction,
        )

        try:
            if llm is not None:
                response, duration = await llm.generate_with_field(
                    system_prompt=MUTATION_SYSTEM_PROMPT,
                    user_input=user_prompt,
                    field_embeddings=field_embeddings,
                    max_new_tokens=1024,
                    temperature=0.7,
                )
            else:
                from llamarcute_live import ollama_client
                response, duration = await ollama_client.chat(
                    system_prompt=MUTATION_SYSTEM_PROMPT,
                    user_message=user_prompt,
                    timeout_sec=120,
                )

            if response is None:
                logger.warning("Mutation %s: LLM returned None", candidate_id)
                continue

            changes = parse_changes(response)
            if changes is None:
                logger.warning("Mutation %s: Failed to parse changes", candidate_id)
                continue

            logger.info(
                "Mutation %s: Parsed %d changes (%.1fs)",
                candidate_id, len(changes), duration,
            )

            mutated, applied = current_personality.apply_changes(
                changes=changes,
                max_changes=max_changes,
                current_ver=next_ver,
            )

            if not applied:
                logger.warning("Mutation %s: No changes applied", candidate_id)
                continue

            mutated._data["id"] = candidate_id
            candidates.append(mutated)

            for desc in applied:
                logger.info("  %s: %s", candidate_id, desc)

        except Exception as e:
            logger.error("Mutation %s failed: %s", candidate_id, e)
            continue

    logger.info(
        "Generated %d/%d mutation candidates",
        len(candidates), num_candidates,
    )
    return candidates
