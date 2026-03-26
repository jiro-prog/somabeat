"""B-3 + B-4: Projection head training with α exploration.

Trains projection_text (384→512→384) and projection_img (768→512→384)
with contrastive loss + structure preservation regularization.

Reads:
  data/vision_phase_b/text_cache.npy
  data/vision_phase_b/img_cache.npy
  data/vision_phase_b/splits.json

Outputs per α:
  data/vision_phase_b/alpha_{α}/projection_text.pt
  data/vision_phase_b/alpha_{α}/projection_img.pt
  data/vision_phase_b/alpha_{α}/training_curves.json
  data/vision_phase_b/alpha_{α}/checkpoint_best.pt
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from shared_state.projection_heads import ProjectionImg, ProjectionText

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "vision_phase_b"


def pearson_corrcoef(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute Pearson correlation between two 1D tensors."""
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    cov = (x_centered * y_centered).sum()
    std_x = x_centered.pow(2).sum().sqrt()
    std_y = y_centered.pow(2).sum().sqrt()
    return cov / (std_x * std_y + 1e-8)


def compute_loss(
    text_emb: torch.Tensor,
    img_emb: torch.Tensor,
    proj_text: ProjectionText,
    proj_img: ProjectionImg,
    tau: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, float, float]:
    """Compute contrastive + structure preservation loss."""
    text_proj = proj_text(text_emb)
    img_proj = proj_img(img_emb)

    # Normalize for cosine similarity
    text_proj_norm = F.normalize(text_proj, dim=-1)
    img_proj_norm = F.normalize(img_proj, dim=-1)

    # InfoNCE
    logits = (text_proj_norm @ img_proj_norm.T) / tau.exp()
    labels = torch.arange(len(logits), device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    loss_contrastive = (loss_i2t + loss_t2i) / 2

    # Structure preservation regularization
    with torch.no_grad():
        original_sim = F.normalize(text_emb, dim=-1) @ F.normalize(text_emb, dim=-1).T
    projected_sim = text_proj_norm @ text_proj_norm.T

    loss_structure = 1.0 - pearson_corrcoef(
        original_sim.flatten(), projected_sim.flatten()
    )

    loss_total = loss_contrastive + alpha * loss_structure

    return loss_total, loss_contrastive.item(), loss_structure.item()


def compute_recall(
    text_emb: np.ndarray,
    img_emb: np.ndarray,
    proj_text: ProjectionText,
    proj_img: ProjectionImg,
    device: str,
    batch_size: int = 1000,
) -> dict:
    """Compute cross-modal retrieval recall on potentially large sets."""
    proj_text.eval()
    proj_img.eval()

    # Project in batches
    all_text_proj = []
    all_img_proj = []
    n = len(text_emb)

    with torch.no_grad():
        for i in range(0, n, batch_size):
            t = torch.from_numpy(text_emb[i:i+batch_size]).to(device)
            im = torch.from_numpy(img_emb[i:i+batch_size]).to(device)
            all_text_proj.append(F.normalize(proj_text(t), dim=-1).cpu().numpy())
            all_img_proj.append(F.normalize(proj_img(im), dim=-1).cpu().numpy())

    text_proj = np.vstack(all_text_proj)
    img_proj = np.vstack(all_img_proj)

    # Compute similarity and recall
    results = {}
    for direction, q, db in [("img2text", img_proj, text_proj),
                              ("text2img", text_proj, img_proj)]:
        # Compute in chunks to avoid OOM
        ks = [1, 5, 10]
        correct = {k: 0 for k in ks}
        chunk = 500
        for i in range(0, n, chunk):
            q_chunk = q[i:i+chunk]
            sim = q_chunk @ db.T  # (chunk, n)
            for k in ks:
                top_k = np.argsort(-sim, axis=1)[:, :k]
                for j in range(len(q_chunk)):
                    if (i + j) in top_k[j]:
                        correct[k] += 1
        for k in ks:
            results[f"{direction}_R@{k}"] = correct[k] / n

    proj_text.train()
    proj_img.train()
    return results


def compute_spearman(
    text_emb: np.ndarray,
    proj_text: ProjectionText,
    device: str,
    n_samples: int = 2000,
) -> float:
    """Compute Spearman rank correlation between original and projected text similarities."""
    from scipy.stats import spearmanr

    # Subsample for efficiency
    rng = np.random.RandomState(42)
    idx = rng.choice(len(text_emb), size=min(n_samples, len(text_emb)), replace=False)
    subset = text_emb[idx]

    # Original similarities
    orig_norm = subset / (np.linalg.norm(subset, axis=1, keepdims=True) + 1e-8)
    orig_sim = orig_norm @ orig_norm.T

    # Projected similarities
    proj_text.eval()
    with torch.no_grad():
        t = torch.from_numpy(subset).to(device)
        proj = F.normalize(proj_text(t), dim=-1).cpu().numpy()
    proj_sim = proj @ proj.T
    proj_text.train()

    # Flatten upper triangle (exclude diagonal)
    mask = np.triu(np.ones_like(orig_sim, dtype=bool), k=1)
    corr, _ = spearmanr(orig_sim[mask], proj_sim[mask])
    return float(corr)


def train_one_alpha(
    alpha: float,
    text_cache: np.ndarray,
    img_cache: np.ndarray,
    splits: dict,
    device: str,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 5,
) -> dict:
    """Train projection heads for one α value."""
    out_dir = DATA_DIR / f"alpha_{alpha}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"TRAINING α={alpha}")
    print(f"{'='*60}")

    # Initialize models
    proj_text = ProjectionText().to(device)
    proj_img = ProjectionImg().to(device)
    log_tau = nn.Parameter(torch.tensor(np.log(0.07), device=device))

    params = list(proj_text.parameters()) + list(proj_img.parameters()) + [log_tau]
    optimizer = torch.optim.AdamW(params, lr=lr)
    total_params = sum(p.numel() for p in params)
    print(f"Parameters: {total_params:,}")

    # Data loaders
    train_idx = splits["train"]
    val_idx = splits["val"]

    train_text = torch.from_numpy(text_cache[train_idx])
    train_img = torch.from_numpy(img_cache[train_idx])
    train_ds = TensorDataset(train_text, train_img)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # Training loop
    curves = {"train_loss": [], "train_contrastive": [], "train_structure": [],
              "val_recall_i2t_10": [], "val_recall_t2i_10": [], "val_spearman": []}
    best_val_score = -1
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        proj_text.train()
        proj_img.train()
        epoch_loss = 0
        epoch_cont = 0
        epoch_struct = 0
        n_batches = 0

        t0 = time.time()
        for text_batch, img_batch in train_loader:
            text_batch = text_batch.to(device)
            img_batch = img_batch.to(device)

            loss, cont, struct = compute_loss(
                text_batch, img_batch, proj_text, proj_img, log_tau, alpha
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_cont += cont
            epoch_struct += struct
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        avg_cont = epoch_cont / n_batches
        avg_struct = epoch_struct / n_batches
        elapsed = time.time() - t0

        curves["train_loss"].append(avg_loss)
        curves["train_contrastive"].append(avg_cont)
        curves["train_structure"].append(avg_struct)

        # Validation every 5 epochs
        if epoch % 5 == 0 or epoch == epochs - 1:
            val_recall = compute_recall(
                text_cache[val_idx], img_cache[val_idx],
                proj_text, proj_img, device
            )
            val_spearman = compute_spearman(text_cache[val_idx], proj_text, device)

            curves["val_recall_i2t_10"].append(val_recall["img2text_R@10"])
            curves["val_recall_t2i_10"].append(val_recall["text2img_R@10"])
            curves["val_spearman"].append(val_spearman)

            # Combined score: average of both R@10 (both must be ≥ 0.5)
            val_score = (val_recall["img2text_R@10"] + val_recall["text2img_R@10"]) / 2

            print(f"  Epoch {epoch:3d}: loss={avg_loss:.4f} (cont={avg_cont:.4f} struct={avg_struct:.4f}) "
                  f"| val R@10 i2t={val_recall['img2text_R@10']:.3f} t2i={val_recall['text2img_R@10']:.3f} "
                  f"| spearman={val_spearman:.3f} | τ={log_tau.exp().item():.4f} | {elapsed:.1f}s")

            if val_score > best_val_score:
                best_val_score = val_score
                patience_counter = 0
                best_state = {
                    "proj_text": proj_text.state_dict(),
                    "proj_img": proj_img.state_dict(),
                    "log_tau": log_tau.item(),
                    "epoch": epoch,
                    "val_score": val_score,
                }
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break
        else:
            print(f"  Epoch {epoch:3d}: loss={avg_loss:.4f} (cont={avg_cont:.4f} struct={avg_struct:.4f}) | {elapsed:.1f}s")

    # Restore best and save
    if best_state:
        proj_text.load_state_dict(best_state["proj_text"])
        proj_img.load_state_dict(best_state["proj_img"])

    torch.save(proj_text.state_dict(), out_dir / "projection_text.pt")
    torch.save(proj_img.state_dict(), out_dir / "projection_img.pt")
    torch.save(best_state, out_dir / "checkpoint_best.pt")
    with open(out_dir / "training_curves.json", "w") as f:
        json.dump(curves, f, indent=2)

    print(f"\n  Best val score: {best_val_score:.4f} at epoch {best_state['epoch']}")
    return {
        "alpha": alpha,
        "best_epoch": best_state["epoch"],
        "best_val_score": best_val_score,
        "final_curves": {k: v[-1] if v else None for k, v in curves.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphas", nargs="+", type=float, default=[1.0, 0.5, 0.1])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    print("Loading cached embeddings...")
    text_cache = np.load(DATA_DIR / "text_cache.npy")
    img_cache = np.load(DATA_DIR / "img_cache.npy")
    with open(DATA_DIR / "splits.json") as f:
        splits = {k: np.array(v) for k, v in json.load(f).items()}

    print(f"Text: {text_cache.shape}, Image: {img_cache.shape}")
    print(f"Splits: {', '.join(f'{k}={len(v)}' for k, v in splits.items())}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    results = []
    for alpha in args.alphas:
        result = train_one_alpha(
            alpha, text_cache, img_cache, splits, device,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        )
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    for r in results:
        c = r["final_curves"]
        print(f"  α={r['alpha']}: best_epoch={r['best_epoch']} "
              f"val_R@10_i2t={c.get('val_recall_i2t_10', '?')} "
              f"val_R@10_t2i={c.get('val_recall_t2i_10', '?')} "
              f"spearman={c.get('val_spearman', '?')}")

    with open(DATA_DIR / "training_summary.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
