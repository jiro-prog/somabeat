"""3-step quality/speed comparison: baseline vs TurboQuant vs FieldReceptor.

Step A: Qwen3-8B 4-bit only (no TQ, no FieldReceptor)
Step B: + TurboQuant 3-bit (no FieldReceptor)
Step C: + FieldReceptor injection (with TurboQuant)

Usage:
  # Kill running bot first (shares GPU)
  pkill -f discord_bot.bot
  .venv/bin/python scripts/compare_quality.py
"""

from __future__ import annotations

import asyncio
import logging
import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from llamarcute_live.personality import Personality
from llamarcute_live.llm_inference import FieldAwareLLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

WARMUP_INPUT = "テスト"
WARMUP_SYSTEM = "日本語で短く返答してください。"


def build_system_prompt() -> str:
    """Same system prompt as DialogueManager.build_prompt()."""
    p = Personality.load("llamarcute_live/data/personality_v0.yaml")
    sections = []
    sections.append("あなたは以下の行動規範に従うAIです。")
    sections.append("")
    sections.append("## 不変制約（自己改善の対象外）")
    sections.append("- 応答言語: 日本語で応答すること")
    sections.append("- 応答長: Discordの2000文字制限を意識し、自然な区切りで収めること")
    sections.append("- 本セクションは自己改善による変更の対象外である")
    sections.append("")
    sections.append("## 行動規範")
    sections.append(p.to_prompt_section())
    return "\n".join(sections)


def get_field_embeddings() -> np.ndarray | None:
    """Get real field embeddings from current shared field state."""
    try:
        from shared_state.backends.chromadb_backend import ChromaDBField
        from shared_state.field_receptor import FieldReceptorImpl
        from shared_state.interface import PerceiveParams

        field = ChromaDBField(persist_directory="data/chromadb")
        receptor = FieldReceptorImpl.load(
            "data/fieldreceptor/field_receptor.pt",
            device="cuda",
        )
        params = PerceiveParams(max_signals=10)

        perception = asyncio.get_event_loop().run_until_complete(
            field.perceive(params),
        )
        if not perception.signals:
            logger.info("  No field signals available")
            return None

        embeddings = receptor.transduce(
            [ps.signal.embedding for ps in perception.signals],
            [ps.strength for ps in perception.signals],
        )
        logger.info("  Field signals: %d → embeddings %s", len(perception.signals), embeddings.shape)
        # Move receptor off GPU to free VRAM for LLM
        receptor.cpu()
        torch.cuda.empty_cache()
        return embeddings
    except Exception as e:
        logger.warning("  FieldReceptor unavailable: %s", e)
        return None


async def run_step(
    label: str,
    llm: FieldAwareLLM,
    system_prompt: str,
    user_input: str,
    field_embeddings: np.ndarray | None = None,
) -> dict:
    """Run one generation step and return results."""
    logger.info("--- %s ---", label)

    # 1. Load model
    llm.load()
    torch.cuda.synchronize()

    # 2. Warmup: short dummy generation to trigger CUDA JIT / kernel caches
    logger.info("  Warmup...")
    await llm.generate_with_field(
        system_prompt=WARMUP_SYSTEM,
        user_input=WARMUP_INPUT,
        field_embeddings=None,
        max_new_tokens=3,
    )
    torch.cuda.synchronize()

    # 3. Reset peak stats AFTER warmup
    torch.cuda.reset_peak_memory_stats()
    vram_before = torch.cuda.memory_allocated() / 1024**2
    logger.info("  Post-warmup VRAM: %.0fMiB", vram_before)

    # 4. Measured generation
    response, duration = await llm.generate_with_field(
        system_prompt=system_prompt,
        user_input=user_input,
        field_embeddings=field_embeddings,
        max_new_tokens=128,
        temperature=0.7,
    )
    torch.cuda.synchronize()

    vram_peak = torch.cuda.max_memory_allocated() / 1024**2
    vram_after = torch.cuda.memory_allocated() / 1024**2

    logger.info("  Response (%d chars, %.1fs):", len(response) if response else 0, duration)
    if response:
        for line in response.split("\n")[:5]:
            logger.info("    > %s", line[:100])
    logger.info(
        "  VRAM: before=%.0fMiB, peak=%.0fMiB, after=%.0fMiB",
        vram_before, vram_peak, vram_after,
    )

    return {
        "label": label,
        "response": response,
        "duration": duration,
        "vram_peak": vram_peak,
        "chars": len(response) if response else 0,
    }


async def main():
    user_input = "最近どんな感じ？"
    system_prompt = build_system_prompt()

    logger.info("System prompt: %d chars", len(system_prompt))
    logger.info("User input: %s", user_input)
    logger.info("")

    results = []

    # === Step A: Baseline (no TQ, no FieldReceptor) ===
    llm_a = FieldAwareLLM(kv_cache_bits=0)
    r = await run_step("Step A: Baseline (no TQ, no FR)", llm_a, system_prompt, user_input)
    results.append(r)
    llm_a.unload()
    torch.cuda.empty_cache()
    logger.info("")

    # === Step B: + TurboQuant 3-bit (no FieldReceptor) ===
    llm_b = FieldAwareLLM(kv_cache_bits=3)
    r = await run_step("Step B: + TurboQuant 3-bit", llm_b, system_prompt, user_input)
    results.append(r)
    llm_b.unload()
    torch.cuda.empty_cache()
    logger.info("")

    # === Step C: + FieldReceptor (with TurboQuant) ===
    logger.info("Loading FieldReceptor...")
    field_embeddings = get_field_embeddings()
    llm_c = FieldAwareLLM(kv_cache_bits=3)
    r = await run_step(
        "Step C: + FieldReceptor", llm_c, system_prompt, user_input,
        field_embeddings=field_embeddings,
    )
    results.append(r)
    llm_c.unload()
    torch.cuda.empty_cache()

    # === Summary ===
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for r in results:
        logger.info(
            "  %-35s  %3d chars  %5.1fs  peak=%5.0fMiB",
            r["label"], r["chars"], r["duration"], r["vram_peak"],
        )
    logger.info("")
    logger.info("=== RESPONSES ===")
    for r in results:
        logger.info("")
        logger.info("[%s]", r["label"])
        logger.info("%s", r["response"][:500] if r["response"] else "(None)")


if __name__ == "__main__":
    asyncio.run(main())
