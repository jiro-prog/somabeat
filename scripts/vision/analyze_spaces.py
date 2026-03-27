"""A-3: Cross-space structural analysis (CKA, Procrustes, PCA).

Reads:
  data/vision_phase_a/text_embeddings.npy  (N, 384)
  data/vision_phase_a/img_embeddings.npy   (N, 768)

Outputs:
  data/vision_phase_a/analysis_report.json
"""

import json
from pathlib import Path

import numpy as np
from scipy.spatial import procrustes
from sklearn.decomposition import PCA

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "vision_phase_a"


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear Centered Kernel Alignment between two representations.

    X: (n, d1), Y: (n, d2) — same n samples.
    Returns CKA in [0, 1]. Higher = more structurally similar.
    """
    n = X.shape[0]
    # Center
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    # Gram matrices (linear kernel)
    K = X @ X.T
    L = Y @ Y.T
    # HSIC
    hsic_kl = np.sum(K * L)
    hsic_kk = np.sum(K * K)
    hsic_ll = np.sum(L * L)
    cka = hsic_kl / (np.sqrt(hsic_kk) * np.sqrt(hsic_ll) + 1e-10)
    return float(cka)


def procrustes_analysis(X: np.ndarray, Y: np.ndarray) -> dict:
    """Orthogonal Procrustes: align Y to X after matching dimensions via PCA.

    Returns disparity (lower = easier to align).
    """
    # Match dimensions to min(n_samples, d1, d2) via PCA
    d_min = min(X.shape[0], X.shape[1], Y.shape[1])
    pca_x = PCA(n_components=d_min)
    pca_y = PCA(n_components=d_min)
    X_r = pca_x.fit_transform(X)
    Y_r = pca_y.fit_transform(Y)
    # Scipy procrustes (standardizes and finds optimal rotation)
    _, _, disparity = procrustes(X_r, Y_r)
    return {
        "disparity": float(disparity),
        "reduced_dim": d_min,
    }


def pca_analysis(X: np.ndarray, name: str, thresholds: list[float] = None) -> dict:
    """PCA variance analysis: effective dimensionality at various thresholds."""
    if thresholds is None:
        thresholds = [0.80, 0.90, 0.95, 0.99]
    pca = PCA().fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    effective_dims = {}
    for t in thresholds:
        n_dims = int(np.searchsorted(cumvar, t) + 1)
        effective_dims[f"{t:.0%}"] = n_dims
    return {
        "name": name,
        "original_dim": X.shape[1],
        "effective_dims": effective_dims,
        "top10_variance_ratio": [float(v) for v in pca.explained_variance_ratio_[:10]],
        "cumulative_at_256": float(cumvar[min(255, len(cumvar) - 1)]),
    }


def recommend_common_dim(text_pca: dict, img_pca: dict) -> dict:
    """Recommend common space dimension D based on PCA results."""
    text_95 = text_pca["effective_dims"]["95%"]
    text_90 = text_pca["effective_dims"]["90%"]
    img_95 = img_pca["effective_dims"]["95%"]
    img_90 = img_pca["effective_dims"]["90%"]

    # Check if 256 covers at least 90% of both spaces
    text_at_256 = text_pca["cumulative_at_256"]
    img_at_256 = img_pca["cumulative_at_256"]

    if img_95 <= 256 and text_95 <= 256:
        recommended_d = 256
        reason = f"Both spaces have 95%-effective dims within 256 (text={text_95}, img={img_95})"
    elif max(text_95, img_95) > 256:
        recommended_d = 384
        reason = (
            f"95%-effective dims exceed 256 (text={text_95}, img={img_95}); "
            f"D=384 covers text@95% and img@~93%. "
            f"Cumulative variance at 256: text={text_at_256:.1%}, img={img_at_256:.1%}"
        )
    else:
        recommended_d = 256
        reason = f"Default: text={text_95}, img={img_95}"

    return {
        "recommended_D": recommended_d,
        "reason": reason,
        "text_95pct_dims": text_95,
        "img_95pct_dims": img_95,
        "text_cumvar_at_256": float(text_at_256),
        "img_cumvar_at_256": float(img_at_256),
    }


def main():
    print("Loading embeddings...")
    text_emb = np.load(DATA_DIR / "text_embeddings.npy")
    img_emb = np.load(DATA_DIR / "img_embeddings.npy")
    print(f"Text: {text_emb.shape}, Image: {img_emb.shape}")

    assert text_emb.shape[0] == img_emb.shape[0], "Sample count mismatch"

    # A-3a: Linear CKA
    print("\nComputing Linear CKA...")
    cka_value = linear_cka(text_emb, img_emb)
    print(f"  CKA = {cka_value:.4f}")

    # A-3b: Procrustes analysis
    print("\nComputing Procrustes analysis...")
    proc_result = procrustes_analysis(text_emb, img_emb)
    print(f"  Disparity = {proc_result['disparity']:.4f}")

    # A-3c: PCA analysis
    print("\nPCA analysis (text space)...")
    text_pca = pca_analysis(text_emb, "e5-small (384d)")
    print(f"  Effective dims: {text_pca['effective_dims']}")

    print("\nPCA analysis (image space)...")
    img_pca = pca_analysis(img_emb, "SigLIP ViT-B (768d)")
    print(f"  Effective dims: {img_pca['effective_dims']}")

    # Dimension recommendation
    dim_rec = recommend_common_dim(text_pca, img_pca)
    print(f"\nRecommended common dim D = {dim_rec['recommended_D']}: {dim_rec['reason']}")

    # Determine if MLP is needed
    mlp_assessment = "linear_sufficient" if cka_value > 0.5 else "mlp_recommended"
    if cka_value < 0.3:
        mlp_layers = "2+ layers"
    elif cka_value < 0.5:
        mlp_layers = "1 hidden layer"
    else:
        mlp_layers = "linear (0 hidden layers)"

    report = {
        "cka": {"value": cka_value, "interpretation": mlp_assessment},
        "procrustes": proc_result,
        "pca_text": text_pca,
        "pca_image": img_pca,
        "dimension_recommendation": dim_rec,
        "mlp_recommendation": {
            "assessment": mlp_assessment,
            "suggested_layers": mlp_layers,
        },
    }

    out_path = DATA_DIR / "analysis_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to {out_path}")

    # Summary
    print("\n" + "=" * 60)
    print("A-3 SUMMARY")
    print("=" * 60)
    print(f"Linear CKA:        {cka_value:.4f} → {mlp_assessment}")
    print(f"Procrustes disp:   {proc_result['disparity']:.4f}")
    print(f"Text 95% dims:     {text_pca['effective_dims']['95%']} / {text_pca['original_dim']}")
    print(f"Image 95% dims:    {img_pca['effective_dims']['95%']} / {img_pca['original_dim']}")
    print(f"Recommended D:     {dim_rec['recommended_D']}")
    print(f"MLP layers:        {mlp_layers}")


if __name__ == "__main__":
    main()
