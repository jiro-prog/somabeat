"""A-4: Linear projection baseline (image→text space).

Trains a linear projection W: img(768) → text(384)
Loss: cosine embedding loss
Evaluates: cross-modal retrieval Recall@1, @5, @10

Reads:
  data/vision_phase_a/text_embeddings.npy  (N, 384)
  data/vision_phase_a/img_embeddings.npy   (N, 768)

Outputs:
  data/vision_phase_a/linear_projection.pt
  data/vision_phase_a/baseline_report.json
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "vision_phase_a"


def split_data(
    text_emb: np.ndarray, img_emb: np.ndarray
) -> tuple[dict, dict, dict]:
    """Split into train(5000) / val(520) / test(500)."""
    n = text_emb.shape[0]
    indices = np.random.RandomState(42).permutation(n)

    train_idx = indices[:5000]
    val_idx = indices[5000:5520]
    test_idx = indices[5520:6020]

    def make_split(idx):
        return {
            "text": torch.from_numpy(text_emb[idx]),
            "img": torch.from_numpy(img_emb[idx]),
        }

    return make_split(train_idx), make_split(val_idx), make_split(test_idx)


class LinearProjection(nn.Module):
    def __init__(self, in_dim: int = 768, out_dim: int = 384):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def compute_recall(
    projected_img: np.ndarray,
    text_emb: np.ndarray,
    ks: list[int] = None,
) -> dict:
    """Cross-modal retrieval recall (image→text and text→image)."""
    if ks is None:
        ks = [1, 5, 10]
    # Normalize
    proj_norm = projected_img / (np.linalg.norm(projected_img, axis=1, keepdims=True) + 1e-8)
    text_norm = text_emb / (np.linalg.norm(text_emb, axis=1, keepdims=True) + 1e-8)

    # Similarity matrix (N x N)
    sim = proj_norm @ text_norm.T
    n = sim.shape[0]

    results = {}
    for direction, sim_matrix in [("img2text", sim), ("text2img", sim.T)]:
        for k in ks:
            top_k = np.argsort(-sim_matrix, axis=1)[:, :k]
            correct = sum(1 for i in range(n) if i in top_k[i])
            results[f"{direction}_R@{k}"] = correct / n

    return results


def train(
    train_data: dict,
    val_data: dict,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cpu",
) -> LinearProjection:
    """Train linear projection with cosine embedding loss."""
    in_dim = train_data["img"].shape[1]
    out_dim = train_data["text"].shape[1]
    model = LinearProjection(in_dim, out_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CosineEmbeddingLoss()

    dataset = TensorDataset(train_data["img"], train_data["text"])
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    # Target: all pairs are positive (matched image-caption)
    best_val_loss = float("inf")
    best_state = model.state_dict().copy()
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for img_batch, text_batch in loader:
            img_batch = img_batch.to(device)
            text_batch = text_batch.to(device)
            target = torch.ones(img_batch.shape[0], device=device)

            projected = model(img_batch)
            loss = loss_fn(projected, text_batch, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * img_batch.shape[0]

        avg_train_loss = total_loss / len(dataset)

        # Validation
        model.eval()
        with torch.no_grad():
            val_proj = model(val_data["img"].to(device))
            val_target = torch.ones(val_data["img"].shape[0], device=device)
            val_loss = loss_fn(val_proj, val_data["text"].to(device), val_target).item()

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}: train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model


def main():
    print("Loading embeddings...")
    text_emb = np.load(DATA_DIR / "text_embeddings.npy")
    img_emb = np.load(DATA_DIR / "img_embeddings.npy")
    print(f"Text: {text_emb.shape}, Image: {img_emb.shape}")

    # A-4a: Split
    print("\nSplitting data (5000/520/500)...")
    train_data, val_data, test_data = split_data(text_emb, img_emb)

    # A-4b: Train linear projection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nTraining linear projection on {device}...")
    print(f"  Parameters: {768 * 384:,} (768→384)")
    model = train(train_data, val_data, device=device)

    # Save model
    torch.save(model.state_dict(), DATA_DIR / "linear_projection.pt")

    # A-4c: Evaluate on test set
    print("\nEvaluating on test set...")
    model.eval()
    with torch.no_grad():
        test_proj = model(test_data["img"].to(device)).cpu().numpy()
    test_text = test_data["text"].numpy()

    recall = compute_recall(test_proj, test_text)
    print("\nRecall results:")
    for k, v in sorted(recall.items()):
        print(f"  {k}: {v:.4f}")

    # Gate check
    r10_img2text = recall["img2text_R@10"]
    gate_pass = r10_img2text >= 0.3
    print(f"\nGate: img2text_R@10 = {r10_img2text:.4f} {'PASS' if gate_pass else 'FAIL'} (threshold: 0.3)")

    report = {
        "split": {"train": 5000, "val": 520, "test": 500},
        "training": {
            "parameters": 768 * 384,
            "device": device,
        },
        "recall": recall,
        "gate": {
            "metric": "img2text_R@10",
            "value": r10_img2text,
            "threshold": 0.3,
            "result": "PASS" if gate_pass else "FAIL",
        },
    }

    out_path = DATA_DIR / "baseline_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
