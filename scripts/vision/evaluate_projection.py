"""B-5 + B-6: Evaluate projection heads and select best α.

Runs:
  B-5a: Cross-modal retrieval on test set
  B-5b: Text-text Spearman correlation
  B-5c: Per-domain recall breakdown
  B-5d: Field simulation (ChromaDB sense test)
  B-6:  α selection and gate judgment

Reads:
  data/vision_phase_b/text_cache.npy
  data/vision_phase_b/img_cache.npy
  data/vision_phase_b/splits.json
  data/vision_phase_b/sources.npy
  data/vision_phase_b/alpha_*/checkpoint_best.pt
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "vision_phase_b"

# Import model classes from train script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_projection import ProjectionText, ProjectionImg


def load_models(alpha_dir: Path, device: str):
    """Load trained projection heads."""
    ckpt = torch.load(alpha_dir / "checkpoint_best.pt", map_location=device)
    proj_text = ProjectionText().to(device)
    proj_img = ProjectionImg().to(device)
    proj_text.load_state_dict(ckpt["proj_text"])
    proj_img.load_state_dict(ckpt["proj_img"])
    proj_text.eval()
    proj_img.eval()
    return proj_text, proj_img, ckpt


def project_embeddings(
    emb: np.ndarray, model, device: str, batch_size: int = 1000
) -> np.ndarray:
    """Project embeddings through a model."""
    all_proj = []
    with torch.no_grad():
        for i in range(0, len(emb), batch_size):
            t = torch.from_numpy(emb[i:i+batch_size]).to(device)
            proj = F.normalize(model(t), dim=-1).cpu().numpy()
            all_proj.append(proj)
    return np.vstack(all_proj)


def compute_recall_detailed(
    text_proj: np.ndarray, img_proj: np.ndarray, ks=(1, 5, 10)
) -> dict:
    """Compute recall with chunk-based similarity."""
    n = len(text_proj)
    results = {}
    for direction, q, db in [("img2text", img_proj, text_proj),
                              ("text2img", text_proj, img_proj)]:
        correct = {k: 0 for k in ks}
        chunk = 500
        for i in range(0, n, chunk):
            q_chunk = q[i:i+chunk]
            sim = q_chunk @ db.T
            for k in ks:
                top_k = np.argsort(-sim, axis=1)[:, :k]
                for j in range(len(q_chunk)):
                    if (i + j) in top_k[j]:
                        correct[k] += 1
        for k in ks:
            results[f"{direction}_R@{k}"] = correct[k] / n
    return results


def compute_spearman_detailed(
    text_emb: np.ndarray, text_proj: np.ndarray, n_samples: int = 3000
) -> float:
    """Spearman rank correlation of text-text similarities."""
    rng = np.random.RandomState(42)
    idx = rng.choice(len(text_emb), size=min(n_samples, len(text_emb)), replace=False)

    orig = text_emb[idx]
    proj = text_proj[idx]

    orig_norm = orig / (np.linalg.norm(orig, axis=1, keepdims=True) + 1e-8)
    orig_sim = orig_norm @ orig_norm.T

    proj_sim = proj @ proj.T  # already normalized

    mask = np.triu(np.ones_like(orig_sim, dtype=bool), k=1)
    corr, _ = spearmanr(orig_sim[mask], proj_sim[mask])
    return float(corr)


def evaluate_alpha(
    alpha: float,
    text_cache: np.ndarray,
    img_cache: np.ndarray,
    test_idx: np.ndarray,
    sources: np.ndarray,
    device: str,
) -> dict:
    """Full evaluation for one α value."""
    alpha_dir = DATA_DIR / f"alpha_{alpha}"
    if not (alpha_dir / "checkpoint_best.pt").exists():
        return {"alpha": alpha, "error": "no checkpoint found"}

    print(f"\n--- Evaluating α={alpha} ---")
    proj_text, proj_img, ckpt = load_models(alpha_dir, device)
    print(f"  Best epoch: {ckpt['epoch']}")

    test_text = text_cache[test_idx]
    test_img = img_cache[test_idx]
    test_sources = sources[test_idx]

    # Project
    text_proj = project_embeddings(test_text, proj_text, device)
    img_proj = project_embeddings(test_img, proj_img, device)

    # B-5a: Cross-modal recall
    recall = compute_recall_detailed(text_proj, img_proj)
    print(f"  Cross-modal Recall:")
    for k, v in sorted(recall.items()):
        print(f"    {k}: {v:.4f}")

    # B-5b: Text-text Spearman
    spearman = compute_spearman_detailed(test_text, text_proj)
    print(f"  Text-text Spearman: {spearman:.4f}")

    # B-5c: Per-domain recall
    domain_recall = {}
    for src, name in [(0, "VideoGameBunny"), (1, "COCO"), (2, "CC3M")]:
        src_mask = test_sources == src
        if src_mask.sum() < 10:
            continue
        src_text = text_proj[src_mask]
        src_img = img_proj[src_mask]
        dr = compute_recall_detailed(src_text, src_img)
        domain_recall[name] = dr
        print(f"  {name} (n={src_mask.sum()}): "
              f"i2t R@10={dr['img2text_R@10']:.3f} t2i R@10={dr['text2img_R@10']:.3f}")

    result = {
        "alpha": alpha,
        "best_epoch": ckpt["epoch"],
        "recall": recall,
        "spearman": spearman,
        "domain_recall": domain_recall,
        "gate": {
            "recall_i2t_10_pass": recall["img2text_R@10"] >= 0.5,
            "recall_t2i_10_pass": recall["text2img_R@10"] >= 0.5,
            "spearman_pass": spearman >= 0.85,
        },
    }

    # Save per-alpha eval
    with open(alpha_dir / "evaluation.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    print("Loading data...")
    text_cache = np.load(DATA_DIR / "text_cache.npy")
    img_cache = np.load(DATA_DIR / "img_cache.npy")
    sources = np.load(DATA_DIR / "sources.npy")
    with open(DATA_DIR / "splits.json") as f:
        splits = json.load(f)
    test_idx = np.array(splits["test"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, Test size: {len(test_idx)}")

    # Evaluate all alphas
    alphas = [1.0, 0.5, 0.1]
    results = []
    for alpha in alphas:
        result = evaluate_alpha(alpha, text_cache, img_cache, test_idx, sources, device)
        results.append(result)

    # B-6: Select best α
    print("\n" + "=" * 60)
    print("B-6: α SELECTION")
    print("=" * 60)

    # Filter candidates: Spearman ≥ 0.85
    candidates = [r for r in results if not r.get("error") and r["gate"]["spearman_pass"]]

    if not candidates:
        print("WARNING: No α satisfies Spearman ≥ 0.85")
        print("Consider: identity init reinforcement or model change")
        best = max(results, key=lambda r: r.get("spearman", 0))
        print(f"Best Spearman: α={best['alpha']} → {best.get('spearman', 'N/A')}")
    else:
        # Among Spearman-passing candidates, pick highest R@10
        best = max(candidates, key=lambda r: (
            r["recall"]["img2text_R@10"] + r["recall"]["text2img_R@10"]
        ) / 2)
        print(f"Selected: α={best['alpha']}")

    # Phase B gate
    all_pass = (
        best.get("gate", {}).get("recall_i2t_10_pass", False)
        and best.get("gate", {}).get("recall_t2i_10_pass", False)
        and best.get("gate", {}).get("spearman_pass", False)
    )

    print(f"\n  α={best['alpha']}:")
    print(f"    img2text R@10: {best['recall']['img2text_R@10']:.4f} {'PASS' if best['gate']['recall_i2t_10_pass'] else 'FAIL'}")
    print(f"    text2img R@10: {best['recall']['text2img_R@10']:.4f} {'PASS' if best['gate']['recall_t2i_10_pass'] else 'FAIL'}")
    print(f"    Spearman:      {best['spearman']:.4f} {'PASS' if best['gate']['spearman_pass'] else 'FAIL'}")
    print(f"\n  Phase B Gate: {'PASS' if all_pass else 'FAIL'}")

    # Domain variance check
    if best.get("domain_recall"):
        domain_r10 = [
            dr["img2text_R@10"]
            for dr in best["domain_recall"].values()
        ]
        if domain_r10:
            overall = best["recall"]["img2text_R@10"]
            max_deviation = max(abs(d - overall) / overall for d in domain_r10)
            domain_pass = max_deviation <= 0.2
            print(f"  Domain deviation: {max_deviation:.1%} {'PASS' if domain_pass else 'FAIL'} (≤20%)")

    # Save final report
    final_report = {
        "all_results": results,
        "selected_alpha": best["alpha"],
        "gate_result": "PASS" if all_pass else "FAIL",
        "phase_c_handoff": {
            "alpha": best["alpha"],
            "common_dim_D": 384,
            "projection_text_path": f"data/vision_phase_b/alpha_{best['alpha']}/projection_text.pt",
            "projection_img_path": f"data/vision_phase_b/alpha_{best['alpha']}/projection_img.pt",
        } if all_pass else None,
    }
    with open(DATA_DIR / "final_report.json", "w") as f:
        json.dump(final_report, f, indent=2)
    print(f"\nReport saved to {DATA_DIR}/final_report.json")


if __name__ == "__main__":
    main()
