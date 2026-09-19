"""C-5: Phase C integration verification.

Runs all manual verification checks:
- C-5a: sense() with re-encoded field
- C-5b: visual signal emit and sense
- C-5c: latency measurements
- C-5d: sleep cycle safety (default disabled)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import yaml

from shared_state.backends.chromadb_backend import ChromaDBField
from shared_state.interface import SenseParams, Signal, SignalOrigin
from shared_state.multimodal_encoder import MultimodalFieldEncoder
from sensory.vision import SensoryVision

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("verify_phase_c")


async def verify_sense_after_reencode(field, encoder):
    """C-5a: Verify sense() works with re-encoded embeddings."""
    logger.info("=== C-5a: sense() after re-encode ===")
    query = encoder.encode_for_sense("最近の自分の調子")
    reading = await field.sense(query, SenseParams(max_signals=10))
    logger.info("Sense returned %d signals", len(reading.signals))
    for ws in reading.signals[:5]:
        logger.info("  [%.3f] %s: %s", ws.effective_weight, ws.signal.origin.context, ws.signal.signal_id[:8])
    assert len(reading.signals) > 0, "sense() returned no signals after re-encode"
    logger.info("C-5a PASS")


async def verify_vision_emit_and_sense(field, encoder):
    """C-5b: Emit a visual signal and verify it exists in the field.

    Note: random dummy images have low cosine similarity to text queries,
    so they won't rank highly in sense() results. This is expected behavior.
    We verify emission via snapshot + origin filter (the correct pattern
    per SharedField design: snapshot() for catalog-style retrieval).
    """
    logger.info("=== C-5b: visual signal emit + sense ===")
    vision = SensoryVision(encoder, field, thalamus_threshold=0.1)

    # Count existing vision signals
    snap_before = await field.snapshot()
    before_count = sum(1 for s in snap_before.signals if s.origin.system == "sensory:vision")

    dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    emitted = await vision.process_frame(dummy_image)
    assert emitted, "First frame should always emit"

    # Verify signal was added
    snap_after = await field.snapshot()
    after_count = sum(1 for s in snap_after.signals if s.origin.system == "sensory:vision")
    logger.info("Vision signals: %d -> %d", before_count, after_count)
    assert after_count > before_count, "Visual signal not found in field after emit"

    # Verify image embedding is retrievable via image query (same-modality sense)
    query_img = encoder.encode_image(dummy_image)
    reading = await field.sense(query_img, SenseParams(max_signals=5, min_relevance=0.0))
    vision_hits = [ws for ws in reading.signals if ws.signal.origin.system == "sensory:vision"]
    logger.info("Image-query sense: %d vision signals in top 5 (relevance=%.4f)",
                len(vision_hits), vision_hits[0].relevance if vision_hits else 0)
    assert len(vision_hits) > 0, "Visual signal not retrievable via image query"

    logger.info("C-5b PASS")

    # Cleanup: purge old test signals by age
    from shared_state.interface import PurgeCriteria
    await field.purge(PurgeCriteria(older_than=timedelta(seconds=1)))


def verify_latency(encoder):
    """C-5c: Measure encode latencies."""
    logger.info("=== C-5c: latency measurements ===")
    results = {}

    # encode_image
    dummy = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    # Warm-up
    encoder.encode_image(dummy)
    t0 = time.perf_counter()
    for _ in range(5):
        encoder.encode_image(dummy)
    elapsed = (time.perf_counter() - t0) / 5
    results["encode_image_ms"] = elapsed * 1000
    logger.info("  encode_image: %.1f ms (target: <500ms)", elapsed * 1000)

    # encode_for_emit
    encoder.encode_for_emit("test warmup")
    t0 = time.perf_counter()
    for _ in range(5):
        encoder.encode_for_emit("テスト文章")
    elapsed = (time.perf_counter() - t0) / 5
    results["encode_for_emit_ms"] = elapsed * 1000
    logger.info("  encode_for_emit: %.1f ms (target: <50ms)", elapsed * 1000)

    # encode_for_sense
    t0 = time.perf_counter()
    for _ in range(5):
        encoder.encode_for_sense("テスト文章")
    elapsed = (time.perf_counter() - t0) / 5
    results["encode_for_sense_ms"] = elapsed * 1000
    logger.info("  encode_for_sense: %.1f ms (target: <50ms)", elapsed * 1000)

    assert results["encode_image_ms"] < 500, f"encode_image too slow: {results['encode_image_ms']:.1f}ms"
    logger.info("C-5c PASS")
    return results


def verify_sleep_config():
    """C-5d: Verify vision is disabled by default."""
    logger.info("=== C-5d: sleep cycle safety ===")
    config_path = Path("config/system.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    vision_enabled = config.get("sensory", {}).get("vision", {}).get("enabled", False)
    logger.info("  sensory.vision.enabled = %s", vision_enabled)
    assert vision_enabled is False, "Vision should be disabled by default"
    logger.info("C-5d PASS")


async def main():
    logger.info("=== Phase C Integration Verification ===")

    # Load config
    with open("config/system.yaml") as f:
        config = yaml.safe_load(f)

    # Setup field and encoder
    field = ChromaDBField(
        persist_directory=config["shared_state"]["chromadb"]["persist_directory"],
        collection_name=config["shared_state"]["chromadb"]["collection_name"],
    )
    encoder = MultimodalFieldEncoder(device="cpu")

    # C-5a
    await verify_sense_after_reencode(field, encoder)

    # C-5b
    await verify_vision_emit_and_sense(field, encoder)

    # C-5c
    latency = verify_latency(encoder)

    # C-5d
    verify_sleep_config()

    logger.info("=== All Phase C Verification Checks PASSED ===")
    logger.info("Latency summary: %s", json.dumps(latency, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
