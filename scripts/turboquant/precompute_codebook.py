#!/usr/bin/env python3
"""Precompute Lloyd-Max codebooks for TurboQuant KV cache compression.

Generates deterministic codebook files (.npz) containing centroids and
boundaries for the Beta(alpha, alpha) distribution. Once generated, these
files are loaded at runtime to guarantee identical quantization across
all compress/decompress operations.

Usage:
    python scripts/turboquant/precompute_codebook.py --dim 128 --bits 2,3,4

Output:
    data/turboquant/codebook_d128_b2.npz
    data/turboquant/codebook_d128_b3.npz
    data/turboquant/codebook_d128_b4.npz

Reference: docs/instructions/instructions_turboquant.md TQ-3
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared_state.turboquant import LloydMaxCodebook

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "turboquant"


def codebook_path(dim: int, bits: int) -> Path:
    """Canonical path for a codebook file."""
    return DATA_DIR / f"codebook_d{dim}_b{bits}.npz"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute Lloyd-Max codebooks for TurboQuant.",
    )
    parser.add_argument(
        "--dim", type=int, default=128,
        help="Vector dimension (= head_dim). Default: 128 (Qwen3-8B)",
    )
    parser.add_argument(
        "--bits", type=str, default="2,3,4",
        help="Comma-separated bit widths to compute. Default: 2,3,4",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing codebook files.",
    )
    args = parser.parse_args()

    dim = args.dim
    bit_list = [int(b.strip()) for b in args.bits.split(",")]

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for bits in bit_list:
        path = codebook_path(dim, bits)

        if path.exists() and not args.force:
            logger.info("SKIP  %s (already exists, use --force to overwrite)", path.name)
            # Verify loadability
            cb = LloydMaxCodebook.load(path)
            logger.info(
                "       dim=%d, bits=%d, levels=%d, centroids[0..2]=%s",
                cb.dim, cb.bits, cb.n_levels,
                cb.centroids[:3].round(6).tolist(),
            )
            continue

        logger.info("Computing %d-bit codebook for dim=%d (%d levels)...", bits, dim, 2**bits)
        t0 = time.monotonic()
        cb = LloydMaxCodebook(dim, bits)
        elapsed = time.monotonic() - t0

        cb.save(path)
        logger.info(
            "SAVED %s (%.2fs, %d centroids, %d boundaries)",
            path.name, elapsed, len(cb.centroids), len(cb.boundaries),
        )

        # Verify roundtrip
        cb_loaded = LloydMaxCodebook.load(path)
        import numpy as np
        assert np.array_equal(cb.centroids, cb_loaded.centroids), "Centroid mismatch after save/load!"
        assert np.array_equal(cb.boundaries, cb_loaded.boundaries), "Boundary mismatch after save/load!"
        logger.info("       Roundtrip verification: OK")

    logger.info("Done. Codebooks saved to %s", DATA_DIR)


if __name__ == "__main__":
    main()
