"""TurboQuant — KV cache compression via online vector quantization.

Implements the TurboQuant algorithm from:
  Zandieh et al., "TurboQuant: Online Vector Quantization with
  Near-optimal Distortion Rate", arXiv:2504.19874, ICLR 2026.

Compresses KV cache to 3-bit/coordinate (~5x reduction) to keep
Qwen3-8B 4-bit inference within 8GB VRAM on RTX 3060 Ti.

Reference: docs/instructions/instructions_turboquant.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# =====================================================================
# TQ-1a: Lloyd-Max Codebook
# =====================================================================

class LloydMaxCodebook:
    """Optimal scalar quantizer for Beta(alpha, alpha) distribution.

    Computes Lloyd-Max centroids and boundaries via iterative expectation
    updates over the Beta density on [-1, 1].
    """

    def __init__(self, dim: int, bits: int, max_iter: int = 200, tol: float = 1e-10):
        self.n_levels = 2 ** bits
        self.alpha = (dim - 1) / 2  # Beta(alpha, alpha) parameter
        self.bits = bits
        self.dim = dim
        self.centroids, self.boundaries = self._compute_codebook(max_iter, tol)
        # Pre-convert to torch for GPU quantization
        self._centroids_torch: torch.Tensor | None = None
        self._boundaries_torch: torch.Tensor | None = None

    def _compute_codebook(
        self, max_iter: int, tol: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Lloyd-Max algorithm on Beta(alpha, alpha) scaled to [-1, 1]."""
        from scipy.stats import beta as beta_dist
        from scipy.integrate import quad

        a = self.alpha
        K = self.n_levels

        # Beta(a, a) on [0, 1] — we work in [0, 1] then map to [-1, 1]
        dist = beta_dist(a, a)

        # Initial centroids: equally spaced quantiles
        quantile_points = np.linspace(0, 1, K + 1)
        centroids = np.array([
            (quantile_points[i] + quantile_points[i + 1]) / 2 for i in range(K)
        ])

        for iteration in range(max_iter):
            # Boundaries: midpoints between adjacent centroids
            boundaries = np.array([
                (centroids[i] + centroids[i + 1]) / 2 for i in range(K - 1)
            ])
            # Full boundary list with endpoints
            full_bounds = np.concatenate([[0.0], boundaries, [1.0]])

            # Update centroids: E[X | t_{j-1} < X < t_j]
            new_centroids = np.empty(K)
            for j in range(K):
                lo, hi = full_bounds[j], full_bounds[j + 1]
                if hi - lo < 1e-15:
                    new_centroids[j] = (lo + hi) / 2
                    continue

                numerator, _ = quad(lambda x: x * dist.pdf(x), lo, hi)
                denominator, _ = quad(dist.pdf, lo, hi)

                if denominator < 1e-15:
                    new_centroids[j] = (lo + hi) / 2
                else:
                    new_centroids[j] = numerator / denominator

            # Convergence check
            shift = np.max(np.abs(new_centroids - centroids))
            centroids = new_centroids
            if shift < tol:
                logger.debug(
                    "Lloyd-Max converged in %d iterations (shift=%.2e)",
                    iteration + 1, shift,
                )
                break

        # Final boundaries
        boundaries = np.array([
            (centroids[i] + centroids[i + 1]) / 2 for i in range(K - 1)
        ])

        # Map from [0, 1] to [-1, 1]
        centroids_scaled = centroids * 2 - 1
        boundaries_scaled = boundaries * 2 - 1

        return centroids_scaled.astype(np.float64), boundaries_scaled.astype(np.float64)

    def _ensure_torch(self, device: torch.device) -> None:
        """Lazily move centroids/boundaries to GPU."""
        if (
            self._centroids_torch is not None
            and self._centroids_torch.device == device
        ):
            return
        self._centroids_torch = torch.from_numpy(
            self.centroids.astype(np.float32)
        ).to(device)
        self._boundaries_torch = torch.from_numpy(
            self.boundaries.astype(np.float32)
        ).to(device)

    def quantize_torch(self, values: torch.Tensor) -> torch.Tensor:
        """Quantize continuous values to codebook indices (GPU)."""
        self._ensure_torch(values.device)
        # searchsorted: find bin index for each value
        indices = torch.searchsorted(self._boundaries_torch, values.contiguous())
        return indices.to(torch.uint8)

    def dequantize_torch(self, indices: torch.Tensor) -> torch.Tensor:
        """Restore centroid values from indices (GPU)."""
        self._ensure_torch(indices.device)
        return self._centroids_torch[indices.long()]

    def quantize(self, values: NDArray) -> NDArray:
        """Quantize continuous values to codebook indices (CPU/numpy)."""
        indices = np.searchsorted(self.boundaries, values)
        return indices.astype(np.uint8)

    def dequantize(self, indices: NDArray) -> NDArray:
        """Restore centroid values from indices (CPU/numpy)."""
        return self.centroids[indices]

    def save(self, path: str | Path) -> None:
        np.savez(
            path,
            centroids=self.centroids,
            boundaries=self.boundaries,
            dim=self.dim,
            bits=self.bits,
        )

    @classmethod
    def load(cls, path: str | Path) -> LloydMaxCodebook:
        data = np.load(path)
        obj = cls.__new__(cls)
        obj.centroids = data["centroids"]
        obj.boundaries = data["boundaries"]
        obj.dim = int(data["dim"])
        obj.bits = int(data["bits"])
        obj.n_levels = 2 ** obj.bits
        obj.alpha = (obj.dim - 1) / 2
        obj._centroids_torch = None
        obj._boundaries_torch = None
        return obj


# =====================================================================
# TQ-1b: Random Rotation Matrix
# =====================================================================

def generate_rotation_matrix(dim: int, seed: int = 42) -> NDArray[np.float32]:
    """Generate a d×d orthogonal matrix (Haar measure) via QR decomposition.

    After rotation, each coordinate z_i follows Beta((d-1)/2, (d-1)/2)
    scaled to [-1, 1], enabling optimal per-coordinate scalar quantization.

    Args:
        dim: Vector dimension (= head_dim, 128 for Qwen3-8B).
        seed: Fixed seed for reproducibility.

    Returns:
        (dim, dim) orthogonal matrix with det = +1.
    """
    rng = np.random.default_rng(seed)
    H = rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(H)
    # Normalize Q so that det = +1
    Q = Q @ np.diag(np.sign(np.diag(R)))
    return Q.astype(np.float32)


# =====================================================================
# TQ-1c: QJL 1-bit Residual Correction
# =====================================================================

def generate_qjl_matrix(
    dim: int, m: int | None = None, seed: int = 43,
) -> NDArray[np.float32]:
    """Generate QJL random projection matrix (Rademacher).

    S ∈ R^{m×d}, each entry ±1/√m.

    Args:
        dim: Vector dimension.
        m: Projection dimension (default: dim).
        seed: Fixed seed.

    Returns:
        (m, dim) random projection matrix.
    """
    if m is None:
        m = dim
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1, 1], size=(m, dim)).astype(np.float32)
    return signs / np.sqrt(m)


class QJLCorrector:
    """QJL 1-bit residual correction for unbiased inner product estimation."""

    def __init__(self, dim: int, seed: int = 43):
        self._S_np = generate_qjl_matrix(dim, seed=seed)
        self._S_torch: torch.Tensor | None = None
        self.dim = dim
        self.m = self._S_np.shape[0]

    def _ensure_torch(self, device: torch.device) -> None:
        if self._S_torch is not None and self._S_torch.device == device:
            return
        self._S_torch = torch.from_numpy(self._S_np).to(device=device, dtype=torch.float32)

    def encode(self, residual: torch.Tensor) -> torch.Tensor:
        """Compress residual to 1-bit signs.

        Args:
            residual: (..., d) tensor.

        Returns:
            bool tensor (..., m).
        """
        self._ensure_torch(residual.device)
        projected = residual.float() @ self._S_torch.T  # (..., m)
        return projected > 0

    def pack_signs(self, signs: torch.Tensor) -> torch.Tensor:
        """Pack bool signs to uint8 (8x memory reduction).

        Args:
            signs: bool tensor, last dim must be divisible by 8.

        Returns:
            uint8 tensor with last dim = signs.shape[-1] // 8.
        """
        shape = signs.shape
        flat = signs.reshape(-1, shape[-1])  # (N, m)
        # Pad to multiple of 8 if needed
        m = flat.shape[1]
        pad = (8 - m % 8) % 8
        if pad > 0:
            flat = torch.nn.functional.pad(flat, (0, pad))
        packed = torch.zeros(
            flat.shape[0], flat.shape[1] // 8,
            dtype=torch.uint8, device=signs.device,
        )
        for bit in range(8):
            packed |= flat[:, bit::8].to(torch.uint8) << bit
        return packed.reshape(*shape[:-1], -1)

    def unpack_signs(self, packed: torch.Tensor, m: int) -> torch.Tensor:
        """Unpack uint8 to bool signs.

        Args:
            packed: uint8 tensor from pack_signs.
            m: Original number of sign bits.

        Returns:
            bool tensor with last dim = m.
        """
        shape = packed.shape
        flat = packed.reshape(-1, shape[-1])  # (N, packed_m)
        unpacked = torch.zeros(
            flat.shape[0], flat.shape[1] * 8,
            dtype=torch.bool, device=packed.device,
        )
        for bit in range(8):
            unpacked[:, bit::8] = (flat >> bit) & 1
        # Trim padding
        unpacked = unpacked[:, :m]
        return unpacked.reshape(*shape[:-1], m)


# =====================================================================
# TQ-1d: Integrated Compressor
# =====================================================================

@dataclass
class CompressedKV:
    """Compressed KV cache data."""
    indices: torch.Tensor       # uint8. (batch, n_heads, seq_len, head_dim)
    norms: torch.Tensor         # float16. (batch, n_heads, seq_len)
    qjl_packed: torch.Tensor    # uint8 bit-packed. (batch, n_heads, seq_len, head_dim // 8)
    residual_norms: torch.Tensor  # float16. (batch, n_heads, seq_len)


class TurboQuantCompressor:
    """TurboQuant KV cache compressor.

    Combines Lloyd-Max codebook quantization with random rotation and
    QJL 1-bit residual correction for near-optimal distortion.

    All operations run on GPU via torch to minimize latency.
    """

    # Default directory for precomputed codebooks (TQ-3)
    _DEFAULT_CODEBOOK_DIR = Path(__file__).resolve().parent.parent / "data" / "turboquant"

    def __init__(
        self,
        head_dim: int = 128,
        bits: int = 3,
        seed: int = 42,
        codebook_path: str | Path | None = None,
    ):
        self.head_dim = head_dim
        self.bits = bits

        # TQ-1a: Codebook — full bits for Lloyd-Max (MSE variant, no QJL)
        # Priority: explicit path > default precomputed file > compute on the fly
        if codebook_path is None:
            default_path = self._DEFAULT_CODEBOOK_DIR / f"codebook_d{head_dim}_b{bits}.npz"
            if default_path.exists():
                codebook_path = default_path

        if codebook_path and Path(codebook_path).exists():
            self.codebook = LloydMaxCodebook.load(codebook_path)
            logger.info("Loaded codebook from %s", codebook_path)
        else:
            logger.warning(
                "Precomputed codebook not found for dim=%d bits=%d. "
                "Computing on the fly (run scripts/turboquant/precompute_codebook.py "
                "to generate deterministic codebooks).",
                head_dim, bits,
            )
            self.codebook = LloydMaxCodebook(head_dim, bits)
            logger.info(
                "Computed %d-bit codebook for dim=%d (%d levels)",
                bits, head_dim, self.codebook.n_levels,
            )

        # TQ-1b: Rotation matrix
        self._rotation_np = generate_rotation_matrix(head_dim, seed=seed)
        self._rotation: torch.Tensor | None = None
        self._rotation_T: torch.Tensor | None = None

        # TQ-1c: QJL corrector
        self.qjl = QJLCorrector(head_dim, seed=seed + 1)

    def _ensure_matrices(self, device: torch.device) -> None:
        """Lazily move rotation matrix to target device."""
        if self._rotation is not None and self._rotation.device == device:
            return
        self._rotation = torch.from_numpy(self._rotation_np).to(device)
        self._rotation_T = self._rotation.T.contiguous()

    def compress(self, kv_tensor: torch.Tensor) -> CompressedKV:
        """Compress KV cache tensor.

        Args:
            kv_tensor: (batch, n_heads, seq_len, head_dim) FP16/BF16.

        Returns:
            CompressedKV with quantized indices, norms, and QJL signs.
        """
        device = kv_tensor.device
        self._ensure_matrices(device)

        # Work in float32 for numerical stability
        x = kv_tensor.float()  # (B, H, S, D)

        # Step 1: Norm extraction
        norms = x.norm(dim=-1, keepdim=True)  # (B, H, S, 1)
        # Avoid division by zero
        safe_norms = norms.clamp(min=1e-8)
        x_hat = x / safe_norms  # Unit sphere

        # Step 2: Random rotation
        z = x_hat @ self._rotation.T  # (B, H, S, D)

        # Step 3: Coordinate-wise Lloyd-Max quantization
        indices = self.codebook.quantize_torch(z)  # uint8
        z_reconstructed = self.codebook.dequantize_torch(indices)  # float32

        # Step 4: Residual
        residual = x_hat - (z_reconstructed @ self._rotation_T.T)  # (B, H, S, D)
        residual_norms = residual.norm(dim=-1)  # (B, H, S)

        # Step 5: QJL 1-bit compression of residual
        # Reshape for QJL: flatten batch dims
        orig_shape = residual.shape  # (B, H, S, D)
        residual_flat = residual.reshape(-1, self.head_dim)  # (B*H*S, D)
        signs = self.qjl.encode(residual_flat)  # (B*H*S, m)
        signs_packed = self.qjl.pack_signs(signs)  # (B*H*S, m//8)
        signs_packed = signs_packed.reshape(
            *orig_shape[:-1], -1,
        )  # (B, H, S, m//8)

        return CompressedKV(
            indices=indices,
            norms=norms.squeeze(-1).half(),
            qjl_packed=signs_packed,
            residual_norms=residual_norms.half(),
        )

    def decompress(self, compressed: CompressedKV) -> torch.Tensor:
        """Decompress KV cache back to FP16.

        Args:
            compressed: CompressedKV from compress().

        Returns:
            (batch, n_heads, seq_len, head_dim) FP16 tensor.
        """
        device = compressed.indices.device
        self._ensure_matrices(device)

        # Dequantize
        z_reconstructed = self.codebook.dequantize_torch(
            compressed.indices,
        )  # float32

        # Inverse rotation
        x_hat = z_reconstructed @ self._rotation_T.T  # (B, H, S, D)

        # Scale by norms
        norms = compressed.norms.float().unsqueeze(-1)  # (B, H, S, 1)
        result = x_hat * norms

        return result.half()

    def memory_bytes(self, compressed: CompressedKV) -> int:
        """Estimate memory usage of compressed KV cache in bytes."""
        total = 0
        total += compressed.indices.numel() * 1  # uint8
        total += compressed.norms.numel() * 2  # float16
        total += compressed.qjl_packed.numel() * 1  # uint8
        total += compressed.residual_norms.numel() * 2  # float16
        return total
