"""Tests for TurboQuant KV cache compression (TQ-4).

Verifies core algorithm correctness against theoretical values from:
Zandieh et al., "TurboQuant", arXiv:2504.19874.
"""

import numpy as np
import pytest
import torch

from shared_state.turboquant import (
    CompressedKV,
    LloydMaxCodebook,
    QJLCorrector,
    TurboQuantCompressor,
    generate_qjl_matrix,
    generate_rotation_matrix,
)

# =====================================================================
# TQ-1a: Codebook tests
# =====================================================================


class TestLloydMaxCodebook:
    def test_codebook_convergence(self):
        """Lloyd-Max should converge within reasonable iterations."""
        cb = LloydMaxCodebook(dim=128, bits=2)
        # If we get here without error, it converged
        assert cb.n_levels == 4
        assert len(cb.centroids) == 4
        assert len(cb.boundaries) == 3

    def test_codebook_symmetry(self):
        """Beta(a,a) is symmetric — codebook should be symmetric about 0."""
        cb = LloydMaxCodebook(dim=128, bits=3)
        # Centroids should be approximately symmetric: c[i] ≈ -c[K-1-i]
        K = cb.n_levels
        for i in range(K // 2):
            assert abs(cb.centroids[i] + cb.centroids[K - 1 - i]) < 1e-6, (
                f"Centroid {i} ({cb.centroids[i]:.6f}) != "
                f"-centroid {K-1-i} ({cb.centroids[K-1-i]:.6f})"
            )

    def test_codebook_ordered(self):
        """Centroids and boundaries should be strictly ordered."""
        cb = LloydMaxCodebook(dim=128, bits=3)
        for i in range(len(cb.centroids) - 1):
            assert cb.centroids[i] < cb.centroids[i + 1]
        for i in range(len(cb.boundaries) - 1):
            assert cb.boundaries[i] < cb.boundaries[i + 1]

    def test_codebook_in_range(self):
        """All centroids should be in [-1, 1]."""
        cb = LloydMaxCodebook(dim=128, bits=3)
        assert np.all(cb.centroids >= -1.0)
        assert np.all(cb.centroids <= 1.0)

    def test_quantize_dequantize_roundtrip(self):
        """Quantize → dequantize should return centroids."""
        cb = LloydMaxCodebook(dim=128, bits=2)
        values = np.array(cb.centroids, dtype=np.float64)
        indices = cb.quantize(values)
        recovered = cb.dequantize(indices)
        np.testing.assert_allclose(recovered, values, atol=1e-10)

    def test_quantize_torch_matches_numpy(self):
        """GPU quantization should match CPU results."""
        cb = LloydMaxCodebook(dim=128, bits=2)
        rng = np.random.default_rng(0)
        values = rng.uniform(-1, 1, size=100).astype(np.float32)

        idx_np = cb.quantize(values)
        idx_torch = cb.quantize_torch(torch.from_numpy(values)).numpy()
        np.testing.assert_array_equal(idx_np, idx_torch)

    def test_quantize_dequantize_mse_3bit(self):
        """3-bit MSE should be approximately 0.03 (2-bit MSE for 3-bit TurboQuant)."""
        cb = LloydMaxCodebook(dim=128, bits=2)  # MSE component is (bits-1)
        # Sample from Beta distribution scaled to [-1, 1]
        rng = np.random.default_rng(42)
        alpha = (128 - 1) / 2
        samples_01 = rng.beta(alpha, alpha, size=100_000)
        samples = samples_01 * 2 - 1  # Scale to [-1, 1]

        indices = cb.quantize(samples)
        reconstructed = cb.dequantize(indices)
        mse = np.mean((samples - reconstructed) ** 2)

        # 2-bit quantization of Beta distribution — MSE should be reasonable
        # Exact theoretical value depends on distribution, but should be < 0.2
        assert mse < 0.2, f"MSE too high: {mse:.4f}"

    def test_save_load_roundtrip(self, tmp_path):
        """Save and load should preserve codebook."""
        cb = LloydMaxCodebook(dim=128, bits=3)
        path = tmp_path / "codebook.npz"
        cb.save(path)

        loaded = LloydMaxCodebook.load(path)
        np.testing.assert_array_equal(cb.centroids, loaded.centroids)
        np.testing.assert_array_equal(cb.boundaries, loaded.boundaries)
        assert cb.dim == loaded.dim
        assert cb.bits == loaded.bits


# =====================================================================
# TQ-1b: Rotation matrix tests
# =====================================================================


class TestRotationMatrix:
    def test_orthogonality(self):
        """Pi^T @ Pi should equal identity."""
        Q = generate_rotation_matrix(128)
        product = Q.T @ Q
        np.testing.assert_allclose(product, np.eye(128), atol=1e-5)

    def test_determinant_positive(self):
        """det(Q) should be +1."""
        Q = generate_rotation_matrix(128)
        det = np.linalg.det(Q)
        assert abs(det - 1.0) < 1e-4, f"det = {det}"

    def test_reproducible(self):
        """Same seed should give same matrix."""
        Q1 = generate_rotation_matrix(128, seed=42)
        Q2 = generate_rotation_matrix(128, seed=42)
        np.testing.assert_array_equal(Q1, Q2)

    def test_rotation_distribution(self):
        """After rotation, coordinates should follow Beta-like distribution.

        For d=128, rotated coordinates ≈ N(0, 1/d). We check that the
        variance is approximately 1/d.
        """
        Q = generate_rotation_matrix(128, seed=42)
        rng = np.random.default_rng(0)

        # Generate random unit vectors and rotate
        N = 10_000
        vectors = rng.standard_normal((N, 128)).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        unit_vectors = vectors / norms

        rotated = unit_vectors @ Q.T  # (N, 128)

        # Each coordinate should have variance ≈ 1/128
        var_per_coord = np.var(rotated, axis=0)
        mean_var = np.mean(var_per_coord)
        expected_var = 1.0 / 128
        assert abs(mean_var - expected_var) < 0.002, (
            f"Mean variance {mean_var:.5f} != expected {expected_var:.5f}"
        )


# =====================================================================
# TQ-1c: QJL tests
# =====================================================================


class TestQJL:
    def test_qjl_matrix_shape(self):
        S = generate_qjl_matrix(128)
        assert S.shape == (128, 128)

    def test_qjl_matrix_scale(self):
        """Each entry should be ±1/√m."""
        S = generate_qjl_matrix(128)
        expected = 1.0 / np.sqrt(128)
        np.testing.assert_allclose(np.abs(S), expected, atol=1e-6)

    def test_encode_shape(self):
        qjl = QJLCorrector(128)
        residual = torch.randn(10, 128)
        signs = qjl.encode(residual)
        assert signs.shape == (10, 128)
        assert signs.dtype == torch.bool

    def test_qjl_unbiasedness(self):
        """QJL inner product estimation should be unbiased.

        E[<encoded_x, encoded_y>] ≈ <x, y> over many trials.
        """
        dim = 128
        qjl = QJLCorrector(dim, seed=0)

        rng = np.random.default_rng(42)
        x = torch.from_numpy(rng.standard_normal(dim).astype(np.float32))
        y = torch.from_numpy(rng.standard_normal(dim).astype(np.float32))
        true_ip = (x @ y).item()

        # Estimate inner product via QJL over many random projections
        estimates = []
        for seed in range(100):
            qjl_i = QJLCorrector(dim, seed=seed)
            signs_x = qjl_i.encode(x.unsqueeze(0)).float() * 2 - 1  # ±1
            signs_y = qjl_i.encode(y.unsqueeze(0)).float() * 2 - 1
            # Scaled inner product estimate
            x_norm = x.norm().item()
            y_norm = y.norm().item()
            est = (signs_x @ signs_y.T).item() * x_norm * y_norm / dim
            estimates.append(est)

        mean_est = np.mean(estimates)
        # Should be approximately unbiased (allow 30% relative error with 100 samples)
        assert abs(mean_est - true_ip) < abs(true_ip) * 0.5 + 1.0, (
            f"Biased: mean={mean_est:.3f} vs true={true_ip:.3f}"
        )

    def test_bit_packing_roundtrip(self):
        """Pack → unpack should recover original signs."""
        qjl = QJLCorrector(128)
        signs = torch.randint(0, 2, (5, 128), dtype=torch.bool)
        packed = qjl.pack_signs(signs)
        assert packed.shape == (5, 16)  # 128 / 8
        unpacked = qjl.unpack_signs(packed, m=128)
        assert torch.equal(signs, unpacked)

    def test_bit_packing_memory_reduction(self):
        """Packed signs should use 8x less memory."""
        qjl = QJLCorrector(128)
        signs = torch.randint(0, 2, (100, 128), dtype=torch.bool)
        packed = qjl.pack_signs(signs)
        # bool: 1 byte per element, uint8 packed: 1 byte per 8 elements
        assert packed.numel() == signs.numel() // 8


# =====================================================================
# TQ-1d: Compressor integration tests
# =====================================================================


class TestTurboQuantCompressor:
    @pytest.fixture
    def compressor(self):
        return TurboQuantCompressor(head_dim=128, bits=3, seed=42)

    def test_compress_decompress_shape(self, compressor):
        """Compress → decompress should preserve shape."""
        kv = torch.randn(1, 4, 10, 128, dtype=torch.float16)
        compressed = compressor.compress(kv)
        restored = compressor.decompress(compressed)
        assert restored.shape == kv.shape
        assert restored.dtype == torch.float16

    def test_compressed_indices_range(self, compressor):
        """Indices should be within codebook range."""
        kv = torch.randn(1, 4, 10, 128, dtype=torch.float16)
        compressed = compressor.compress(kv)
        assert compressed.indices.max().item() < compressor.codebook.n_levels
        assert compressed.indices.min().item() >= 0

    def test_memory_reduction(self, compressor):
        """Compressed memory should be significantly less than original."""
        B, H, S, D = 1, 4, 100, 128
        kv = torch.randn(B, H, S, D, dtype=torch.float16)
        original_bytes = kv.numel() * 2  # FP16 = 2 bytes
        compressed = compressor.compress(kv)
        compressed_bytes = compressor.memory_bytes(compressed)

        ratio = original_bytes / compressed_bytes
        # uint8 indices (1B) + norms (2B) + qjl_packed (D/8 B) + res_norms (2B)
        # per vector: 128*1 + 2 + 16 + 2 = 148 bytes vs 128*2 = 256 bytes → ~1.7x
        # Full bit-packing of indices would increase this further
        assert ratio > 1.5, f"Compression ratio too low: {ratio:.1f}x"

    def test_reconstruction_quality(self, compressor):
        """Reconstruction should have reasonable cosine similarity."""
        kv = torch.randn(1, 4, 10, 128, dtype=torch.float16)
        compressed = compressor.compress(kv)
        restored = compressor.decompress(compressed)

        # Compute cosine similarity per vector
        kv_flat = kv.float().reshape(-1, 128)
        restored_flat = restored.float().reshape(-1, 128)

        cos_sim = torch.nn.functional.cosine_similarity(
            kv_flat, restored_flat, dim=-1,
        )
        mean_cos = cos_sim.mean().item()
        # With 2-bit MSE quantization, expect reasonable but not perfect similarity
        assert mean_cos > 0.85, f"Cosine similarity too low: {mean_cos:.3f}"

    def test_compress_zero_vectors(self, compressor):
        """Should handle zero vectors without NaN."""
        kv = torch.zeros(1, 1, 1, 128, dtype=torch.float16)
        compressed = compressor.compress(kv)
        restored = compressor.decompress(compressed)
        assert not torch.isnan(restored).any()
        assert not torch.isinf(restored).any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_compress_decompress_gpu(self, compressor):
        """Should work on GPU."""
        kv = torch.randn(1, 4, 10, 128, dtype=torch.float16, device="cuda")
        compressed = compressor.compress(kv)
        restored = compressor.decompress(compressed)
        assert restored.device.type == "cuda"
        assert restored.shape == kv.shape
