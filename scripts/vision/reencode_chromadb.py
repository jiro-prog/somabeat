"""C-2: Re-encode existing ChromaDB signals with MultimodalFieldEncoder.

Reads all signals from the shared field, re-encodes their trace text through
projection_text, and updates the embeddings in place. Metadata is preserved.

Usage:
    cd /home/jiro/integrated-system
    .venv/bin/python scripts/vision/reencode_chromadb.py
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import chromadb
import numpy as np

from shared_state.multimodal_encoder import MultimodalFieldEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("reencode_chromadb")

CHROMADB_DIR = "data/chromadb"
COLLECTION_NAME = "shared_field"
BACKUP_PATH = "data/chromadb_backup_pre_vision.json"
BATCH_SIZE = 100


def backup_collection(collection: chromadb.Collection, backup_path: str) -> int:
    """Backup all signal metadata (no embeddings) to JSON."""
    all_data = collection.get(include=["metadatas", "documents"])
    records = []
    for sid, meta, doc in zip(all_data["ids"], all_data["metadatas"], all_data["documents"]):
        records.append({"id": sid, "metadata": meta, "document": doc})

    Path(backup_path).parent.mkdir(parents=True, exist_ok=True)
    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump({"backed_up_at": datetime.now().isoformat(), "count": len(records), "signals": records}, f, ensure_ascii=False, indent=2)

    logger.info("Backup saved: %s (%d signals)", backup_path, len(records))
    return len(records)


def reencode(collection: chromadb.Collection, encoder: MultimodalFieldEncoder) -> int:
    """Re-encode all signals and update embeddings in place."""
    all_data = collection.get(include=["metadatas", "documents", "embeddings"])
    ids = all_data["ids"]
    metadatas = all_data["metadatas"]
    documents = all_data["documents"]
    total = len(ids)
    logger.info("Re-encoding %d signals...", total)

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch_ids = ids[start:end]
        batch_docs = documents[start:end]

        new_embeddings = []
        new_metadatas = []
        for doc, meta in zip(batch_docs, metadatas[start:end]):
            vec = encoder.encode_for_emit(doc)
            new_norm = float(np.linalg.norm(vec))
            new_embeddings.append(vec.tolist())
            # Update norm in metadata, preserve everything else
            updated_meta = dict(meta)
            updated_meta["norm"] = new_norm
            new_metadatas.append(updated_meta)

        collection.update(
            ids=batch_ids,
            embeddings=new_embeddings,
            metadatas=new_metadatas,
        )
        logger.info("  re-encoded %d/%d", end, total)

    return total


def verify(collection: chromadb.Collection, encoder: MultimodalFieldEncoder, backup_path: str) -> bool:
    """Verify re-encode: count unchanged, metadata preserved, embeddings match."""
    with open(backup_path, "r", encoding="utf-8") as f:
        backup = json.load(f)

    expected_count = backup["count"]
    actual_count = collection.count()
    if actual_count != expected_count:
        logger.error("Count mismatch: expected %d, got %d", expected_count, actual_count)
        return False

    # Spot-check 10 random signals
    all_data = collection.get(include=["metadatas", "documents", "embeddings"])
    indices = np.random.choice(len(all_data["ids"]), size=min(10, len(all_data["ids"])), replace=False)

    for idx in indices:
        sid = all_data["ids"][idx]
        doc = all_data["documents"][idx]
        meta = all_data["metadatas"][idx]
        stored_emb = np.array(all_data["embeddings"][idx], dtype=np.float32)

        # Verify metadata preserved (check key fields from backup)
        backup_signal = next((s for s in backup["signals"] if s["id"] == sid), None)
        if backup_signal is None:
            logger.error("Signal %s missing from backup", sid)
            return False

        for key in ["origin_system", "origin_context", "trace", "emitted_at"]:
            if meta.get(key) != backup_signal["metadata"].get(key):
                logger.error("Metadata mismatch for %s.%s: %s vs %s", sid, key, meta.get(key), backup_signal["metadata"].get(key))
                return False

        # Verify embedding matches fresh encode
        expected_emb = encoder.encode_for_emit(doc)
        # ChromaDB normalizes embeddings internally, so compare normalized versions
        stored_norm = stored_emb / (np.linalg.norm(stored_emb) + 1e-10)
        expected_norm = expected_emb / (np.linalg.norm(expected_emb) + 1e-10)
        cos_sim = np.dot(stored_norm, expected_norm)
        if cos_sim < 0.999:
            logger.error("Embedding mismatch for %s: cosine=%.6f", sid, cos_sim)
            return False

    logger.info("Verification passed: %d signals, 10 spot-checks OK", actual_count)
    return True


def main():
    logger.info("=== ChromaDB Re-encode Start ===")

    # 1. Open collection
    client = chromadb.PersistentClient(path=CHROMADB_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)
    count = collection.count()
    logger.info("Collection '%s': %d signals", COLLECTION_NAME, count)

    if count == 0:
        logger.info("No signals to re-encode. Done.")
        return

    # 2. Backup
    backup_count = backup_collection(collection, BACKUP_PATH)
    assert backup_count == count

    # 3. Load encoder
    logger.info("Loading MultimodalFieldEncoder...")
    encoder = MultimodalFieldEncoder(device="cpu")

    # 4. Re-encode
    reencode(collection, encoder)

    # 5. Verify
    ok = verify(collection, encoder, BACKUP_PATH)
    if not ok:
        logger.error("Verification FAILED!")
        sys.exit(1)

    logger.info("=== ChromaDB Re-encode Complete ===")


if __name__ == "__main__":
    main()
