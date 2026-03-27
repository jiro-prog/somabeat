"""Fused decode attention with TurboQuant compressed keys.

Key insight: Q @ K_restored^T = Q @ (norm * Π^T @ z)^T
  = norm * (Q @ Π^T) @ z = norm * q_rotated @ z

By pre-rotating Q (q_rot = Q @ Π^T), the fused attention only
needs codebook lookup + dot product — no inverse rotation matrix multiply.

Initially planned as a Triton JIT kernel, but scalar indexing constraints
in Triton made the codebook-lookup loop impractical. Implemented with
PyTorch GPU ops (torch.matmul + advanced indexing) instead. Profiling
confirmed fused attention is only 6% of total decode time, so the
performance difference is negligible.

Reference: docs/instructions/instructions_turboquant.md TQ-6a
"""

from __future__ import annotations

import math

import torch


def turboquant_decode_attention(
    query: torch.Tensor,
    compressed_keys_indices: torch.Tensor,
    compressed_keys_norms: torch.Tensor,
    codebook_centroids: torch.Tensor,
    rotation_matrix: torch.Tensor,
    values: torch.Tensor,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Fused decode attention with compressed keys.

    Pre-rotates query, dequantizes keys via codebook lookup (no inverse
    rotation needed), computes attention scores, and returns output.

    Uses PyTorch ops on GPU — avoids per-token decompression overhead of
    the full inverse rotation.
    """
    B, n_heads, _, head_dim = query.shape
    _, n_kv_heads, seq_len, _ = values.shape
    gqa_ratio = n_heads // n_kv_heads

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    q = query.float()  # (B, n_heads, 1, D)
    rot = rotation_matrix.float()  # (D, D)

    # Pre-rotate query: q_rot = q @ Π^T (inverse rotation)
    q_rot = q @ rot.T  # (B, n_heads, 1, D)

    # Dequantize all keys in rotated space (batch codebook lookup)
    indices_long = compressed_keys_indices.long()  # (B, n_kv, S, D)
    z = codebook_centroids[indices_long]  # (B, n_kv, S, D) float32
    norms = compressed_keys_norms.float().unsqueeze(-1)  # (B, n_kv, S, 1)

    # Expand KV heads for GQA: (B, n_kv, S, D) → (B, n_heads, S, D)
    z_expanded = z.repeat_interleave(gqa_ratio, dim=1)  # (B, n_heads, S, D)
    norms_expanded = norms.repeat_interleave(gqa_ratio, dim=1)  # (B, n_heads, S, 1)

    # Attention scores: q_rot @ z^T * norm * scale
    # (B, n_heads, 1, D) @ (B, n_heads, D, S) → (B, n_heads, 1, S)
    scores = torch.matmul(q_rot, z_expanded.transpose(-2, -1))
    scores = scores * norms_expanded.transpose(-2, -1) * sm_scale

    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)  # (B, n_heads, 1, S)

    # Value aggregation (expand values for GQA)
    v = values.float().repeat_interleave(gqa_ratio, dim=1)  # (B, n_heads, S, D)
    output = torch.matmul(attn_weights, v)  # (B, n_heads, 1, D)

    return output.half()


def reference_decode_attention(
    query: torch.Tensor,
    compressed_keys_indices: torch.Tensor,
    compressed_keys_norms: torch.Tensor,
    codebook_centroids: torch.Tensor,
    rotation_matrix: torch.Tensor,
    values: torch.Tensor,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """PyTorch reference — loop-based for correctness verification."""
    B, n_heads, _, head_dim = query.shape
    _, n_kv_heads, seq_len, _ = values.shape
    gqa_ratio = n_heads // n_kv_heads

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    q = query.float()
    rot = rotation_matrix.float()
    q_rot = q @ rot.T

    outputs = []
    for b in range(B):
        head_outputs = []
        for h in range(n_heads):
            kv_h = h // gqa_ratio
            indices = compressed_keys_indices[b, kv_h].long()
            z = codebook_centroids[indices]
            norms_vec = compressed_keys_norms[b, kv_h].float()

            qr = q_rot[b, h, 0]
            scores = (z @ qr) * norms_vec * sm_scale
            attn_weights = torch.softmax(scores, dim=0)

            v = values[b, kv_h].float()
            out = (attn_weights.unsqueeze(-1) * v).sum(dim=0)
            head_outputs.append(out)

        outputs.append(torch.stack(head_outputs, dim=0))

    result = torch.stack(outputs, dim=0).unsqueeze(2)
    return result.half()
