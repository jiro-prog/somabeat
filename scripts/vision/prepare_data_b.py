"""B-1 + B-2: Download multi-domain datasets, compute embeddings (streaming).

Streaming design: images are never held in memory. Each batch is:
  1. Streamed from HuggingFace
  2. Encoded by SigLIP → 768d embedding
  3. Written to pre-allocated npy file
  4. Image discarded

Memory footprint: ~2GB (SigLIP model + e5 model + embedding arrays on disk)

Datasets:
  - GameplayCaptions: 50K from asgaardlab/GameplayCaptions (game screens)
  - COCO-Caption: 50K from lmms-lab/COCO-Caption (natural images)
  - CC3M-WDS: 100K from pixparse/cc3m-wds (web images)

Outputs:
  data/vision_phase_b/text_cache.npy      (N, 384)
  data/vision_phase_b/img_cache.npy       (N, 768)
  data/vision_phase_b/metadata.json
  data/vision_phase_b/splits.json
  data/vision_phase_b/captions.json
  data/vision_phase_b/sources.npy
"""

import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoProcessor

SEED = 42
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "vision_phase_b"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GAME_SIZE = 50_000
COCO_SIZE = 50_000
CC3M_SIZE = 100_000
TOTAL_SIZE = GAME_SIZE + COCO_SIZE + CC3M_SIZE  # 200K

IMG_BATCH = 32  # images per SigLIP forward pass
IMG_DIM = 768
TEXT_DIM = 384

BAD_CAPTION_PATTERNS = [
    "a photo of a photo", "image may contain", "no description",
    "stock photo", "getty images", "shutterstock",
]


class StreamingEmbedder:
    """Manages SigLIP model and writes image embeddings to a pre-allocated npy."""

    def __init__(self, out_path: Path, total: int, device: str):
        self.device = device
        self.out_path = out_path
        self.total = total

        # Pre-allocate output file (memory-mapped)
        self._mmap = np.lib.format.open_memmap(
            str(out_path), mode="w+", dtype=np.float32, shape=(total, IMG_DIM),
        )
        self.write_idx = 0
        self._buffer_imgs = []
        self._model = None
        self._processor = None

    def _load_model(self):
        if self._model is not None:
            return
        print(f"  Loading SigLIP ViT-B on {self.device}...")
        model = AutoModel.from_pretrained("google/siglip2-base-patch16-256").to(self.device)
        self._processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-256")
        self._model = model.vision_model
        self._model.eval()

    def add_image(self, img):
        """Buffer one image. Flush when batch is full."""
        self._buffer_imgs.append(img)
        if len(self._buffer_imgs) >= IMG_BATCH:
            self._flush()

    def _flush(self):
        """Encode buffered images and write to mmap."""
        if not self._buffer_imgs:
            return
        self._load_model()
        batch = self._buffer_imgs
        self._buffer_imgs = []

        try:
            inputs = self._processor(images=batch, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self._model(pixel_values=inputs["pixel_values"])
                embs = outputs.pooler_output.cpu().numpy()
        except Exception as e:
            print(f"  Warning: batch at idx {self.write_idx} failed ({e}), zeros")
            embs = np.zeros((len(batch), IMG_DIM), dtype=np.float32)

        n = embs.shape[0]
        self._mmap[self.write_idx : self.write_idx + n] = embs
        self.write_idx += n

        # Free GPU cache periodically
        if self.write_idx % (IMG_BATCH * 50) == 0 and self.device == "cuda":
            torch.cuda.empty_cache()

    def finalize(self):
        """Flush remaining buffer and clean up."""
        self._flush()
        self._mmap.flush()
        actual = self.write_idx
        if actual < self.total:
            # Truncate to actual size
            print(f"  Truncating img_cache from {self.total} to {actual}")
            full = np.load(str(self.out_path), mmap_mode="r")[:actual].copy()
            np.save(str(self.out_path), full)
            del full
        del self._mmap, self._model, self._processor
        self._model = None
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return actual


def stream_gameplay(embedder: StreamingEmbedder, rng: np.random.RandomState) -> list[str]:
    """Stream GameplayCaptions, encode images, return captions."""
    print("\n=== GameplayCaptions (streaming) ===")
    ds = load_dataset("asgaardlab/GameplayCaptions", split="validation", streaming=True)

    # We need to subsample 50K from 76K. Pre-decide which indices to keep.
    # Since streaming doesn't support random access, use reservoir sampling.
    captions = []
    count = 0
    t0 = time.time()

    for item in ds:
        if len(captions) >= GAME_SIZE:
            break
        cap = item.get("blip2-opt-6.7b_captions.csv", "")
        img = item.get("image")
        if not cap or img is None:
            continue

        captions.append(cap)
        embedder.add_image(img)
        count += 1
        if count % 10000 == 0:
            elapsed = time.time() - t0
            print(f"  [{count}/{GAME_SIZE}] {elapsed:.0f}s")

    print(f"  GameplayCaptions: {len(captions)} in {time.time() - t0:.0f}s")
    return captions


def stream_coco(embedder: StreamingEmbedder, rng: np.random.RandomState) -> list[str]:
    """Stream COCO-Caption, encode images, return captions."""
    print("\n=== COCO-Caption (streaming) ===")
    captions = []
    t0 = time.time()

    for split in ["val", "test"]:
        if len(captions) >= COCO_SIZE:
            break
        ds = load_dataset("lmms-lab/COCO-Caption", split=split, streaming=True)
        for item in ds:
            if len(captions) >= COCO_SIZE:
                break
            cap_list = item.get("answer", [])
            if isinstance(cap_list, list) and cap_list:
                cap = cap_list[0]
            elif isinstance(cap_list, str):
                cap = cap_list
            else:
                continue
            if not cap or len(cap) < 5:
                continue
            img = item.get("image")
            if img is None:
                continue

            captions.append(cap)
            embedder.add_image(img)
            if len(captions) % 10000 == 0:
                elapsed = time.time() - t0
                print(f"  [{len(captions)}/{COCO_SIZE}] {elapsed:.0f}s")

    print(f"  COCO: {len(captions)} in {time.time() - t0:.0f}s")
    return captions


def stream_cc3m(embedder: StreamingEmbedder, rng: np.random.RandomState) -> list[str]:
    """Stream CC3M-WDS, encode images, return captions."""
    print("\n=== CC3M-WDS (streaming) ===")
    captions = []
    filtered = 0
    t0 = time.time()

    ds = load_dataset("pixparse/cc3m-wds", split="train", streaming=True)
    for item in ds:
        if len(captions) >= CC3M_SIZE:
            break
        cap = item.get("txt", "")
        img = item.get("jpg")
        if not cap or len(cap) < 10 or img is None:
            filtered += 1
            continue
        if any(p in cap.lower() for p in BAD_CAPTION_PATTERNS):
            filtered += 1
            continue

        captions.append(cap)
        embedder.add_image(img)
        if len(captions) % 20000 == 0:
            elapsed = time.time() - t0
            print(f"  [{len(captions)}/{CC3M_SIZE}] {elapsed:.0f}s (filtered {filtered})")

    print(f"  CC3M: {len(captions)} in {time.time() - t0:.0f}s (filtered {filtered})")
    return captions


def create_splits(n: int, sources: np.ndarray, rng: np.random.RandomState) -> dict:
    """Create train/val/test splits with balanced test set."""
    test_indices = []
    for src, count in [(0, 2500), (1, 2500), (2, 5000)]:
        src_indices = np.where(sources == src)[0]
        chosen = rng.choice(src_indices, size=min(count, len(src_indices)), replace=False)
        test_indices.extend(chosen.tolist())

    test_set = set(test_indices)
    remaining = [i for i in range(n) if i not in test_set]
    rng.shuffle(remaining)

    val_size = min(10000, len(remaining) // 10)
    return {
        "train": remaining[val_size:],
        "val": remaining[:val_size],
        "test": test_indices,
    }


def main():
    rng = np.random.RandomState(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("B-1 + B-2: STREAMING DATA PREP + EMBEDDING")
    print(f"Device: {device}, Target: {TOTAL_SIZE} samples")
    print("=" * 60)

    # Phase 1: Stream images → SigLIP embeddings (written to disk)
    embedder = StreamingEmbedder(OUT_DIR / "img_cache.npy", TOTAL_SIZE, device)

    game_caps = stream_gameplay(embedder, rng)
    coco_caps = stream_coco(embedder, rng)
    cc3m_caps = stream_cc3m(embedder, rng)

    actual_img = embedder.finalize()
    print(f"\nImage embeddings written: {actual_img}")

    # Merge captions and sources
    all_captions = game_caps + coco_caps + cc3m_caps
    sources = np.array(
        [0] * len(game_caps) + [1] * len(coco_caps) + [2] * len(cc3m_caps),
        dtype=np.int8,
    )
    total = len(all_captions)
    print(f"Total captions: {total}")
    print(f"  Game: {len(game_caps)}, COCO: {len(coco_caps)}, CC3M: {len(cc3m_caps)}")

    # Verify img count matches caption count
    if actual_img != total:
        print(f"WARNING: img count ({actual_img}) != caption count ({total})")
        # Truncate to smaller
        total = min(actual_img, total)
        all_captions = all_captions[:total]
        sources = sources[:total]

    # Caption samples
    print("\n--- Caption samples (3 per source) ---")
    for src, name in [(0, "Game"), (1, "COCO"), (2, "CC3M")]:
        idx = np.where(sources == src)[0][:3]
        print(f"{name}:")
        for i in idx:
            print(f"  [{i}] {all_captions[i][:100]}")

    # Phase 2: Text embeddings (captions are just strings, fits in RAM)
    print("\n" + "=" * 60)
    print("B-2a: TEXT EMBEDDINGS")
    print("=" * 60)
    e5 = SentenceTransformer("intfloat/multilingual-e5-small")
    prefixed = [f"passage: {cap}" for cap in all_captions]
    t0 = time.time()
    text_emb = e5.encode(prefixed, show_progress_bar=True, batch_size=128)
    text_emb = np.array(text_emb, dtype=np.float32)
    print(f"Text embeddings: {text_emb.shape} in {time.time() - t0:.1f}s")
    np.save(OUT_DIR / "text_cache.npy", text_emb)
    del e5, prefixed
    gc.collect()

    # Splits
    splits = create_splits(total, sources, rng)
    print(f"\nSplits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    # Save metadata
    np.save(OUT_DIR / "sources.npy", sources)
    with open(OUT_DIR / "captions.json", "w") as f:
        json.dump(all_captions, f, ensure_ascii=False)
    with open(OUT_DIR / "splits.json", "w") as f:
        json.dump({k: [int(i) for i in v] for k, v in splits.items()}, f)
    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump({
            "total": total,
            "sources": {
                "game (GameplayCaptions)": len(game_caps),
                "coco (lmms-lab/COCO-Caption)": len(coco_caps),
                "cc3m (pixparse/cc3m-wds)": len(cc3m_caps),
            },
            "seed": SEED,
            "splits": {k: len(v) for k, v in splits.items()},
        }, f, indent=2)

    # Verification
    print("\n--- Verification ---")
    img_cache = np.load(str(OUT_DIR / "img_cache.npy"), mmap_mode="r")
    print(f"Text: {text_emb.shape}, Image: {img_cache.shape}")
    norms = np.linalg.norm(img_cache[:100], axis=1)
    print(f"Image L2 norm (first 100): mean={norms.mean():.3f}")
    print(f"Text L2 norm (first 100): mean={np.linalg.norm(text_emb[:100], axis=1).mean():.3f}")
    zero_count = np.sum(np.linalg.norm(img_cache, axis=1) < 0.01)
    if zero_count > 0:
        print(f"WARNING: {zero_count} zero image embeddings")

    print(f"\nAll saved to {OUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
