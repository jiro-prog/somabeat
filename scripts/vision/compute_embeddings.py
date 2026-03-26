"""A-2: Compute text and image embeddings for GameplayCaptions.

Dataset: asgaardlab/GameplayCaptions (75,979 samples, validation split only).
We subsample 6,020 for Phase A as specified in the task flow.
Caption source: blip2-opt-6.7b (best quality among available models).

Outputs:
  data/vision_phase_a/text_embeddings.npy  (6020, 384)
  data/vision_phase_a/img_embeddings.npy   (6020, 768)
  data/vision_phase_a/captions.json        caption texts for reference
  data/vision_phase_a/sample_indices.npy   indices into original dataset
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoProcessor

SAMPLE_SIZE = 6020
CAPTION_COL = "blip2-opt-6.7b_captions.csv"
SEED = 42

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "vision_phase_a"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_text_embeddings(captions: list[str]) -> np.ndarray:
    """Encode captions with e5-small using 'passage:' prefix."""
    print(f"Loading e5-small...")
    model = SentenceTransformer("intfloat/multilingual-e5-small")
    prefixed = [f"passage: {cap}" for cap in captions]
    print(f"Encoding {len(prefixed)} captions...")
    t0 = time.time()
    embeddings = model.encode(prefixed, show_progress_bar=True, batch_size=64)
    elapsed = time.time() - t0
    print(f"Text embeddings: {embeddings.shape} in {elapsed:.1f}s")
    return np.array(embeddings, dtype=np.float32)


def compute_image_embeddings(dataset, device: str) -> np.ndarray:
    """Encode images with SigLIP ViT-B."""
    print(f"Loading SigLIP ViT-B on {device}...")
    model = AutoModel.from_pretrained("google/siglip2-base-patch16-256").to(device)
    processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-256")
    vision_model = model.vision_model
    vision_model.eval()

    all_embeddings = []
    batch_size = 32
    n = len(dataset)
    print(f"Encoding {n} images (batch_size={batch_size})...")
    t0 = time.time()

    for i in range(0, n, batch_size):
        batch = dataset[i : min(i + batch_size, n)]
        images = batch["image"]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = vision_model(pixel_values=inputs["pixel_values"])
            pooled = outputs.pooler_output
        all_embeddings.append(pooled.cpu().numpy())

        if (i // batch_size) % 10 == 0:
            print(f"  [{i}/{n}] ...")

    elapsed = time.time() - t0
    result = np.vstack(all_embeddings).astype(np.float32)
    print(f"Image embeddings: {result.shape} in {elapsed:.1f}s")
    return result


def main():
    # Load dataset (should be cached from A-1b)
    print("Loading GameplayCaptions...")
    ds = load_dataset("asgaardlab/GameplayCaptions")
    full = ds["validation"]  # only split available
    print(f"Full dataset: {len(full)} samples, columns: {full.column_names}")

    # Subsample 6,020 for Phase A
    rng = np.random.RandomState(SEED)
    indices = rng.choice(len(full), size=SAMPLE_SIZE, replace=False)
    indices.sort()
    subset = full.select(indices)
    print(f"Subsampled: {len(subset)} samples (seed={SEED})")
    np.save(OUT_DIR / "sample_indices.npy", indices)

    captions = list(subset[CAPTION_COL])
    print(f"Caption column: '{CAPTION_COL}', sample: {captions[0][:100]}")

    # Save captions and metadata
    with open(OUT_DIR / "captions.json", "w") as f:
        json.dump(captions, f, ensure_ascii=False)
    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump({
            "dataset": "asgaardlab/GameplayCaptions",
            "split": "validation",
            "full_size": len(full),
            "sample_size": SAMPLE_SIZE,
            "caption_model": CAPTION_COL,
            "seed": SEED,
            "games": sorted(set(subset["game"])),
        }, f, indent=2, ensure_ascii=False)

    # A-2a: Text embeddings
    text_emb = compute_text_embeddings(captions)
    np.save(OUT_DIR / "text_embeddings.npy", text_emb)

    # A-2b: Image embeddings (GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_emb = compute_image_embeddings(subset, device)
    np.save(OUT_DIR / "img_embeddings.npy", img_emb)

    print(f"\nSaved to {OUT_DIR}/")
    print(f"  text_embeddings.npy: {text_emb.shape}")
    print(f"  img_embeddings.npy:  {img_emb.shape}")
    print(f"  captions.json:       {len(captions)} entries")
    print(f"  sample_indices.npy:  {len(indices)} indices")
    print(f"  metadata.json:       dataset info")


if __name__ == "__main__":
    main()
