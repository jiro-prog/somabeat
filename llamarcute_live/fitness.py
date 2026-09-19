"""Fitness evaluation engine — evaluates personality candidates on task sets.

Each candidate is given a task set (core + rotation) and scored on accuracy.

Reference: llamarcute_live_design.md section 5.4, phase1.5_phase2_taskflow.md T19
         T24: parallel inference with semaphore control
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from shared_state.encoder import E5SmallEncoder

from llamarcute_live.llm_inference import FieldAwareLLM
from llamarcute_live.parser import grade_code, grade_logic, grade_math
from llamarcute_live.personality import Personality

logger = logging.getLogger(__name__)

TASK_SYSTEM_PROMPT = """\
You are an AI assistant. Follow these strategy rules strictly:

{rules}

Solve the following task. Think step by step, then provide your final answer.

For math problems, end with: ANSWER: <number>
For logic/choice problems, end with: ANSWER: <letter>
For coding problems, provide a Python function in a ```python``` code block.
"""


async def evaluate_fitness(
    candidates: list[Personality],
    core_tasks: list[dict],
    rotation_tasks: list[dict],
    encoder: E5SmallEncoder | None = None,
    llm: FieldAwareLLM | None = None,
    parallel: int = 2,
) -> dict[str, dict]:
    """Evaluate each candidate on core tasks + rotation tasks.

    Args:
        candidates: List of Personality instances (including the current one).
        core_tasks: Core benchmark tasks (math, code, logic).
        rotation_tasks: Q&A rotation tasks from the shared field.
        encoder: E5SmallEncoder for scoring rotation tasks by cosine similarity.
        llm: FieldAwareLLM instance for generate_bare inference.
        parallel: Max concurrent inference calls (semaphore limit).

    Returns:
        {personality_id: {
            "core_results": [...],
            "by_domain": {...},
            "core_accuracy": float,
            "rotation_score": float,
            "combined_fitness": float,
        }}
    """
    if llm is None or not llm.is_loaded():
        logger.error("FieldAwareLLM not available — skipping fitness evaluation")
        return {
            c.id: {
                "core_results": [],
                "by_domain": {},
                "core_accuracy": 0.0,
                "rotation_score": 0.0,
                "combined_fitness": 0.0,
            }
            for c in candidates
        }

    try:
        return await _evaluate_fitness_parallel(
            candidates, core_tasks, rotation_tasks,
            encoder, llm, parallel,
        )
    except Exception as e:
        if parallel > 1:
            logger.warning(
                "Parallel evaluation failed (parallel=%d): %s. "
                "Retrying with parallel=1 (OOM fallback).",
                parallel, e,
            )
            return await _evaluate_fitness_parallel(
                candidates, core_tasks, rotation_tasks,
                encoder, llm, 1,
            )
        raise


async def _evaluate_fitness_parallel(
    candidates: list[Personality],
    core_tasks: list[dict],
    rotation_tasks: list[dict],
    encoder: E5SmallEncoder | None,
    llm: FieldAwareLLM,
    parallel: int,
) -> dict[str, dict]:
    """Inner implementation that runs all candidate x task evaluations in parallel."""
    semaphore = asyncio.Semaphore(parallel)
    logger.info(
        "Fitness evaluation: %d candidates, %d core tasks, %d rotation tasks, parallel=%d",
        len(candidates), len(core_tasks), len(rotation_tasks), parallel,
    )

    # --- Build all core task coroutines ---
    # Each coroutine returns (candidate_id, result_dict)
    core_coroutines = []
    for candidate in candidates:
        system_prompt = TASK_SYSTEM_PROMPT.format(rules=candidate.to_prompt_section())
        for task in core_tasks:
            core_coroutines.append(
                _evaluate_single_core_task(
                    semaphore, candidate.id, system_prompt, task, llm,
                )
            )

    # --- Build all rotation task coroutines ---
    # Each coroutine returns (candidate_id, score_float)
    rotation_coroutines = []
    if rotation_tasks and encoder is not None:
        for candidate in candidates:
            system_prompt = TASK_SYSTEM_PROMPT.format(rules=candidate.to_prompt_section())
            for task in rotation_tasks:
                rotation_coroutines.append(
                    _evaluate_single_rotation_task(
                        semaphore, candidate.id, system_prompt, task, encoder, llm,
                    )
                )

    # --- Gather everything with overall timeout ---
    all_coroutines = core_coroutines + rotation_coroutines
    try:
        all_results = await asyncio.wait_for(
            asyncio.gather(*all_coroutines, return_exceptions=True),
            timeout=1800,  # 30 min hard cap for all fitness evaluations
        )
    except asyncio.TimeoutError:
        logger.error("Fitness evaluation timed out after 1800s (%d tasks)", len(all_coroutines))
        # Return zero scores for all candidates
        return {
            c.id: {
                "core_results": [],
                "by_domain": {},
                "core_accuracy": 0.0,
                "rotation_score": 0.0,
                "combined_fitness": 0.0,
            }
            for c in candidates
        }

    core_results_list = all_results[:len(core_coroutines)]
    rotation_results_list = all_results[len(core_coroutines):]

    # --- Aggregate core results by candidate ---
    core_by_candidate: dict[str, list[dict]] = {c.id: [] for c in candidates}
    for result in core_results_list:
        if isinstance(result, Exception):
            logger.error("Core task coroutine raised exception: %s", result)
            continue
        cid, task_result = result
        core_by_candidate[cid].append(task_result)

    # --- Aggregate rotation scores by candidate ---
    rotation_by_candidate: dict[str, list[float]] = {c.id: [] for c in candidates}
    for result in rotation_results_list:
        if isinstance(result, Exception):
            logger.error("Rotation task coroutine raised exception: %s", result)
            continue
        cid, score = result
        rotation_by_candidate[cid].append(score)

    # --- Assemble final results ---
    results = {}
    for candidate in candidates:
        core_results = core_by_candidate[candidate.id]

        # Domain aggregation
        by_domain: dict[str, dict] = {}
        for domain in ["math", "code", "logic"]:
            domain_results = [r for r in core_results if r["domain"] == domain]
            if domain_results:
                score_sum = sum(r["score"] for r in domain_results)
                by_domain[domain] = {
                    "correct": sum(1 for r in domain_results if r["correct"]),
                    "score_sum": score_sum,
                    "total": len(domain_results),
                    "accuracy": score_sum / len(domain_results),
                }

        core_accuracy = (
            sum(r["score"] for r in core_results) / len(core_results)
            if core_results else 0.0
        )

        # Rotation score
        rot_scores = rotation_by_candidate[candidate.id]
        rotation_score = sum(rot_scores) / len(rot_scores) if rot_scores else 0.0

        # Combined: 70% core + 30% rotation (rotation may be empty)
        if rotation_tasks:
            combined = 0.7 * core_accuracy + 0.3 * rotation_score
        else:
            combined = core_accuracy

        results[candidate.id] = {
            "core_results": core_results,
            "by_domain": by_domain,
            "core_accuracy": core_accuracy,
            "rotation_score": rotation_score,
            "combined_fitness": combined,
        }

        logger.info(
            "  %s: core=%.2f rotation=%.2f combined=%.2f",
            candidate.id, core_accuracy, rotation_score, combined,
        )

    return results


async def _evaluate_single_core_task(
    semaphore: asyncio.Semaphore,
    candidate_id: str,
    system_prompt: str,
    task: dict,
    llm: FieldAwareLLM,
) -> tuple[str, dict]:
    """Evaluate a single core task for one candidate, with semaphore control.

    Returns:
        (candidate_id, result_dict)
    """
    domain = task["domain"]

    async with semaphore:
        try:
            response, duration = await llm.generate_bare(
                system_prompt=system_prompt,
                user_input=task["question"],
                max_new_tokens=1024,
            )

            if response is None:
                return candidate_id, {
                    "task_id": task["id"],
                    "domain": domain,
                    "correct": False,
                    "score": 0.0,
                    "error": "LLM returned None",
                }

            # Grade based on domain
            if domain == "math":
                correct = grade_math(response, float(task["answer"]))
                score = 1.0 if correct else 0.0
            elif domain == "logic":
                correct = grade_logic(response, task["answer"])
                score = 1.0 if correct else 0.0
            elif domain == "code":
                grade_result = grade_code(response, task.get("test_cases", []))
                score = grade_result["score"]
                correct = score >= 1.0
            else:
                correct = False
                score = 0.0

            return candidate_id, {
                "task_id": task["id"],
                "domain": domain,
                "correct": correct,
                "score": score,
                "duration": duration,
            }

        except Exception as e:
            logger.error("Task %s error for %s: %s", task["id"], candidate_id, e)
            return candidate_id, {
                "task_id": task["id"],
                "domain": domain,
                "correct": False,
                "score": 0.0,
                "error": str(e),
            }


async def _evaluate_single_rotation_task(
    semaphore: asyncio.Semaphore,
    candidate_id: str,
    system_prompt: str,
    task: dict,
    encoder: E5SmallEncoder,
    llm: FieldAwareLLM,
) -> tuple[str, float]:
    """Evaluate a single rotation task for one candidate, with semaphore control.

    Returns:
        (candidate_id, score)
    """
    instruction = task.get("instruction", "")
    expected_output = task.get("output", "")
    if not instruction or not expected_output:
        return candidate_id, 0.0

    async with semaphore:
        try:
            response, _ = await llm.generate_bare(
                system_prompt=system_prompt,
                user_input=instruction,
                max_new_tokens=1024,
            )

            if response is None:
                return candidate_id, 0.0

            # Cosine similarity between response and expected
            resp_emb = encoder.encode_for_emit(response)
            expected_emb = encoder.encode_for_emit(expected_output)

            # Normalize for cosine similarity
            resp_norm = resp_emb / (np.linalg.norm(resp_emb) + 1e-10)
            exp_norm = expected_emb / (np.linalg.norm(expected_emb) + 1e-10)
            similarity = float(np.dot(resp_norm, exp_norm))

            # Clamp to [0, 1]
            return candidate_id, max(0.0, similarity)

        except Exception as e:
            logger.error("Rotation task scoring error for %s: %s", candidate_id, e)
            return candidate_id, 0.0
