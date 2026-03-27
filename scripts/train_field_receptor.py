"""T5: Train FieldReceptor — field embedding (384d) → LLM embedding space (4096d).

Uses paired data from T4:
  data/fieldreceptor/field_embeddings.npy   (N, 384)
  data/fieldreceptor/llm_A_embed_mean.npy   (N, 4096)

Loss: MSE between FieldReceptor output and LLM embedding layer output.
Validation: cosine similarity between transformed and target vectors.

Outputs:
  data/fieldreceptor/field_receptor.pt        (trained weights)
  data/fieldreceptor/training_report.json     (curves + metrics)

Usage:
  python scripts/train_field_receptor.py [--epochs 200] [--lr 1e-3] [--hidden-dim 512]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared_state.field_receptor import FieldReceptorImpl

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "fieldreceptor"


def cosine_sim_batch(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean cosine similarity between corresponding rows."""
    a_norm = F.normalize(a, dim=-1)
    b_norm = F.normalize(b, dim=-1)
    return float((a_norm * b_norm).sum(dim=-1).mean())


def main():
    parser = argparse.ArgumentParser(description="T5: Train FieldReceptor")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args()

    # 1. Load data
    print("Loading T4 data...")
    X = np.load(DATA_DIR / "field_embeddings.npy")      # (N, 384)
    Y = np.load(DATA_DIR / "llm_A_embed_mean.npy")      # (N, 4096)
    print(f"  X: {X.shape}, Y: {Y.shape}")

    N = X.shape[0]
    field_dim = X.shape[1]
    agent_dim = Y.shape[1]

    # 2. Train/val split (shuffled)
    rng = np.random.RandomState(42)
    indices = rng.permutation(N)
    n_val = max(1, int(N * args.val_ratio))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    X_train = torch.from_numpy(X[train_idx]).float()
    Y_train = torch.from_numpy(Y[train_idx]).float()
    X_val = torch.from_numpy(X[val_idx]).float()
    Y_val = torch.from_numpy(Y[val_idx]).float()

    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

    train_ds = TensorDataset(X_train, Y_train)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    # 3. Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FieldReceptorImpl(
        field_dim=field_dim,
        agent_dim=agent_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    print(f"  Model: {field_dim} → {args.hidden_dim} → {agent_dim}, device={device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 4. Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_cosine_sim": []}

    X_val_d = X_val.to(device)
    Y_val_d = Y_val.to(device)

    t_start = time.monotonic()

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        train_loss = epoch_loss / len(train_idx)

        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_d)
            val_loss = F.mse_loss(val_pred, Y_val_d).item()
            val_cos = cosine_sim_batch(val_pred, Y_val_d)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_cosine_sim"].append(val_cos)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), DATA_DIR / "field_receptor.pt")
        else:
            patience_counter += 1

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{args.epochs}: "
                f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
                f"val_cos={val_cos:.4f}  lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if patience_counter >= args.patience:
            print(f"  Early stopping at epoch {epoch} (patience={args.patience})")
            break

    elapsed = time.monotonic() - t_start
    print(f"\nTraining completed in {elapsed:.1f}s")

    # 5. Final evaluation with best model
    model.load_state_dict(torch.load(DATA_DIR / "field_receptor.pt", weights_only=True))
    model.eval()

    with torch.no_grad():
        # Full dataset evaluation
        X_all_d = torch.from_numpy(X).float().to(device)
        Y_all_d = torch.from_numpy(Y).float().to(device)
        pred_all = model(X_all_d)

        final_mse = F.mse_loss(pred_all, Y_all_d).item()
        final_cos = cosine_sim_batch(pred_all, Y_all_d)

        # Per-sample cosine similarities
        per_sample_cos = (
            F.normalize(pred_all, dim=-1) * F.normalize(Y_all_d, dim=-1)
        ).sum(dim=-1)
        cos_min = float(per_sample_cos.min())
        cos_max = float(per_sample_cos.max())
        cos_median = float(per_sample_cos.median())

    print(f"\n{'='*60}")
    print("T5 TRAINING RESULTS")
    print(f"{'='*60}")
    print(f"  Final MSE (all):        {final_mse:.6f}")
    print(f"  Cosine similarity:      mean={final_cos:.4f}  median={cos_median:.4f}")
    print(f"                          min={cos_min:.4f}  max={cos_max:.4f}")
    print(f"  Best val loss:          {best_val_loss:.6f}")
    print(f"  Epochs trained:         {len(history['train_loss'])}")
    print(f"  Weights saved to:       {DATA_DIR / 'field_receptor.pt'}")
    print(f"{'='*60}")

    # 6. Save report
    report = {
        "architecture": {
            "field_dim": field_dim,
            "hidden_dim": args.hidden_dim,
            "agent_dim": agent_dim,
            "type": "mlp_1hidden",
        },
        "training": {
            "samples_total": N,
            "samples_train": len(train_idx),
            "samples_val": len(val_idx),
            "epochs_trained": len(history["train_loss"]),
            "best_val_loss": best_val_loss,
            "elapsed_seconds": round(elapsed, 1),
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
        },
        "evaluation": {
            "mse": final_mse,
            "cosine_sim_mean": final_cos,
            "cosine_sim_median": cos_median,
            "cosine_sim_min": cos_min,
            "cosine_sim_max": cos_max,
        },
        "history": history,
    }

    report_path = DATA_DIR / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
