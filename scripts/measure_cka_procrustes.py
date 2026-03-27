"""T4: CKA/Procrustes measurement — field embedding vs LLM internal representation.

Measures structural alignment between the shared field's embedding space (384d)
and Qwen3-8B's internal representations to determine FieldReceptor architecture.

Three LLM representation candidates:
  A. Embedding layer output, mean pooling over tokens
  B. Embedding layer output, last token only
  C. Transformer layer 1 output, mean pooling over tokens

Prerequisites:
  - Ollama stopped (to free VRAM)
  - Qwen/Qwen3-8B weights in HF cache
  - bitsandbytes installed (for 4-bit quantization)

Usage:
  python scripts/measure_cka_procrustes.py [--min-samples 50] [--max-samples 200]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "data" / "fieldreceptor"

# Reuse existing CKA/Procrustes from vision analysis
from scripts.vision.analyze_spaces import linear_cka, procrustes_analysis


# ---------------------------------------------------------------------------
# 1. Data collection — dialogue texts from SQLite
# ---------------------------------------------------------------------------

async def load_dialogue_texts(
    db_path: Path, max_samples: int = 200,
) -> list[str]:
    """Load dialogue texts from llamarcute_live's dialogue_log."""
    import aiosqlite

    async with aiosqlite.connect(db_path) as db:
        # Combine user+assistant pairs into single texts for richer semantics
        cursor = await db.execute(
            """SELECT content FROM dialogue_log
               WHERE length(content) > 10
               ORDER BY created_at DESC LIMIT ?""",
            (max_samples * 2,),
        )
        rows = await cursor.fetchall()

    texts = [row[0] for row in rows]
    logger.info("Loaded %d dialogue texts from %s", len(texts), db_path)
    return texts


# ---------------------------------------------------------------------------
# 2. Field embeddings (X) — via E5SmallEncoder
# ---------------------------------------------------------------------------

def compute_field_embeddings(texts: list[str]) -> np.ndarray:
    """Encode texts using the field's E5SmallEncoder."""
    from shared_state.encoder import E5SmallEncoder

    encoder = E5SmallEncoder()
    embeddings = []
    for i, text in enumerate(texts):
        emb = encoder.encode_for_emit(text)
        embeddings.append(emb)
        if (i + 1) % 50 == 0:
            logger.info("  Field encoding: %d/%d", i + 1, len(texts))

    X = np.stack(embeddings, axis=0)
    logger.info("Field embeddings: %s", X.shape)
    return X


# ---------------------------------------------------------------------------
# 3. LLM internal representations (Y) — via transformers + 4-bit
# ---------------------------------------------------------------------------

def load_qwen3_model():
    """Load Qwen3-8B with 4-bit quantization."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_name = "Qwen/Qwen3-8B"
    logger.info("Loading %s with 4-bit quantization...", model_name)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    logger.info("Model loaded. Device: %s", next(model.parameters()).device)
    return model, tokenizer


def compute_llm_representations(
    texts: list[str],
    model,
    tokenizer,
    batch_size: int = 4,
) -> dict[str, np.ndarray]:
    """Extract 3 candidate representations from Qwen3-8B.

    Returns:
        {
            "A_embed_mean": (N, hidden_size),  # embedding layer, mean pool
            "B_embed_last": (N, hidden_size),  # embedding layer, last token
            "C_layer1_mean": (N, hidden_size), # transformer layer 1, mean pool
        }
    """
    import torch

    all_A = []
    all_B = []
    all_C = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        device = next(model.parameters()).device
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        # hidden_states[0] = embedding layer output (before any transformer layer)
        # hidden_states[1] = after transformer layer 0
        # hidden_states[2] = after transformer layer 1

        attention_mask = inputs["attention_mask"]  # (B, T)
        mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, T, 1)

        # A: Embedding layer mean pool
        emb_out = hidden_states[0].float()  # (B, T, H)
        A = (emb_out * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)

        # B: Embedding layer last token
        lengths = attention_mask.sum(dim=1) - 1  # last valid index
        B = emb_out[torch.arange(len(batch_texts)), lengths]

        # C: Transformer layer 1 output mean pool
        layer1_out = hidden_states[2].float()  # [2] = after layer 1 (0-indexed)
        C = (layer1_out * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)

        all_A.append(A.cpu().numpy())
        all_B.append(B.cpu().numpy())
        all_C.append(C.cpu().numpy())

        if (start + batch_size) % 20 == 0 or start + batch_size >= len(texts):
            logger.info("  LLM encoding: %d/%d", min(start + batch_size, len(texts)), len(texts))

    result = {
        "A_embed_mean": np.concatenate(all_A, axis=0),
        "B_embed_last": np.concatenate(all_B, axis=0),
        "C_layer1_mean": np.concatenate(all_C, axis=0),
    }
    for k, v in result.items():
        logger.info("  %s: %s", k, v.shape)

    return result


# ---------------------------------------------------------------------------
# 4. Measurement and decision
# ---------------------------------------------------------------------------

def measure_all(
    X: np.ndarray,
    llm_reps: dict[str, np.ndarray],
) -> dict:
    """Compute CKA and Procrustes for each candidate."""
    results = {}

    for name, Y in llm_reps.items():
        logger.info("Measuring %s (field %s vs LLM %s)...", name, X.shape, Y.shape)
        cka = linear_cka(X, Y)
        proc = procrustes_analysis(X, Y)

        results[name] = {
            "cka": cka,
            "procrustes_disparity": proc["disparity"],
            "field_dim": X.shape[1],
            "llm_dim": Y.shape[1],
        }
        logger.info("  CKA=%.4f  Procrustes_disparity=%.4f", cka, proc["disparity"])

    return results


def make_decision(results: dict) -> dict:
    """Determine best representation and transformation architecture.

    Decision table from instructions:
      CKA high + Procrustes small → linear transform
      CKA high + Procrustes large → MLP
      CKA low → MLP + sufficient training data
    """
    # Find best candidate by CKA
    best_name = max(results, key=lambda k: results[k]["cka"])
    best = results[best_name]

    cka = best["cka"]
    disp = best["procrustes_disparity"]

    if cka > 0.5 and disp < 0.3:
        architecture = "linear"
        reason = f"CKA={cka:.3f} (high), Procrustes={disp:.3f} (small) → linear sufficient"
    elif cka > 0.3:
        architecture = "mlp_1hidden"
        reason = f"CKA={cka:.3f} (moderate), Procrustes={disp:.3f} → 1-hidden MLP"
    else:
        architecture = "mlp_2hidden"
        reason = f"CKA={cka:.3f} (low) → 2-hidden MLP with sufficient training data"

    return {
        "best_candidate": best_name,
        "architecture": architecture,
        "reason": reason,
        "cka": cka,
        "procrustes_disparity": disp,
        "field_dim": best["field_dim"],
        "llm_dim": best["llm_dim"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="T4: CKA/Procrustes measurement")
    parser.add_argument("--min-samples", type=int, default=50)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--db-path", type=str, default=str(PROJECT_ROOT / "data" / "llamarcute_live.db"))
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    # 1. Load texts
    logger.info("=== T4: CKA/Procrustes Measurement ===")
    texts = asyncio.run(load_dialogue_texts(Path(args.db_path), args.max_samples))

    if len(texts) < args.min_samples:
        logger.error(
            "Insufficient data: %d texts (min %d required). "
            "Run more dialogue cycles first.",
            len(texts), args.min_samples,
        )
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Field embeddings (cache to disk)
    field_cache = OUTPUT_DIR / "field_embeddings.npy"
    if field_cache.exists():
        logger.info("Loading cached field embeddings from %s", field_cache)
        X = np.load(field_cache)
    else:
        logger.info("\n--- Computing field embeddings ---")
        X = compute_field_embeddings(texts)
        np.save(field_cache, X)
        logger.info("Saved field embeddings to %s", field_cache)

    # 3. LLM representations (cache to disk)
    llm_cache_A = OUTPUT_DIR / "llm_A_embed_mean.npy"
    if llm_cache_A.exists():
        logger.info("Loading cached LLM representations from %s", OUTPUT_DIR)
        llm_reps = {
            "A_embed_mean": np.load(OUTPUT_DIR / "llm_A_embed_mean.npy"),
            "B_embed_last": np.load(OUTPUT_DIR / "llm_B_embed_last.npy"),
            "C_layer1_mean": np.load(OUTPUT_DIR / "llm_C_layer1_mean.npy"),
        }
    else:
        logger.info("\n--- Loading LLM and computing representations ---")
        model, tokenizer = load_qwen3_model()
        llm_reps = compute_llm_representations(texts, model, tokenizer, args.batch_size)

        # Save immediately before any further processing
        for name, Y in llm_reps.items():
            np.save(OUTPUT_DIR / f"llm_{name}.npy", Y)
        logger.info("Saved LLM representations to %s", OUTPUT_DIR)

        # Free GPU memory
        import torch
        del model
        del tokenizer
        torch.cuda.empty_cache()
        logger.info("GPU memory freed.")

    # 4. Measure
    logger.info("\n--- Computing CKA and Procrustes ---")
    results = measure_all(X, llm_reps)

    # 5. Decision
    decision = make_decision(results)

    # 6. Save report
    report = {
        "samples": len(texts),
        "candidates": results,
        "decision": decision,
    }

    report_path = OUTPUT_DIR / "cka_procrustes_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("\nReport saved to %s", report_path)

    # Summary
    print("\n" + "=" * 60)
    print("T4 MEASUREMENT RESULTS")
    print("=" * 60)
    for name, r in results.items():
        marker = " ★" if name == decision["best_candidate"] else ""
        print(f"  {name}: CKA={r['cka']:.4f}  Procrustes={r['procrustes_disparity']:.4f}{marker}")
    print(f"\nBest candidate:  {decision['best_candidate']}")
    print(f"Architecture:    {decision['architecture']}")
    print(f"Reason:          {decision['reason']}")
    print(f"Dimensions:      field={decision['field_dim']} → LLM={decision['llm_dim']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
