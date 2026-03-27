"""TurboQuantCache — compressed KV cache with fused decode attention.

Compresses keys after RoPE via TurboQuant. During decode (seq_len=1),
uses pre-rotated query optimization to avoid full key decompression.
Values remain in FP16.

Strategy:
- Prefill: store keys compressed + values FP16. Return decompressed
  keys for normal attention computation.
- Decode: store new key compressed, use fused attention for Q@K^T
  (pre-rotate Q, codebook lookup, no inverse rotation).
  Return dummy keys to attention_interface which won't be used
  because we monkey-patch the attention forward.

Reference: docs/instructions/instructions_turboquant.md TQ-2a, TQ-6b
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers.cache_utils import CacheLayerMixin, DynamicCache

from shared_state.turboquant import TurboQuantCompressor

logger = logging.getLogger(__name__)


class TurboQuantLayer(CacheLayerMixin):
    """Cache layer that compresses keys via TurboQuant.

    Stores:
    - Compressed prefill keys (indices + norms)
    - Compressed decode keys (appended per-token)
    - FP16 values (uncompressed, cat'd normally)

    On decode, provides compressed key data for fused attention.
    """

    is_sliding = False

    def __init__(self, compressor: TurboQuantCompressor):
        super().__init__()
        self._compressor = compressor
        # Compressed key storage
        self._key_indices: torch.Tensor | None = None  # (B, n_kv, S, D) uint8
        self._key_norms: torch.Tensor | None = None    # (B, n_kv, S) fp16
        # FP16 storage
        self._fp16_keys: torch.Tensor | None = None     # prefill keys (FP16)
        self._values: torch.Tensor | None = None        # all values (FP16)
        self._seq_length = 0
        self._prefill_done = False

    def lazy_initialization(
        self, key_states: torch.Tensor, value_states: torch.Tensor,
    ) -> None:
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        new_len = key_states.shape[-2]
        self._seq_length += new_len

        # Compress new keys
        compressed = self._compressor.compress(key_states)
        if self._key_indices is None:
            self._key_indices = compressed.indices
            self._key_norms = compressed.norms
        else:
            self._key_indices = torch.cat(
                [self._key_indices, compressed.indices], dim=-2,
            )
            self._key_norms = torch.cat(
                [self._key_norms, compressed.norms], dim=-1,
            )

        # Values: always store FP16 uncompressed
        if self._values is None:
            self._values = value_states
        else:
            self._values = torch.cat(
                [self._values, value_states], dim=-2,
            )

        if new_len > 1:
            # Prefill: return ORIGINAL FP16 keys for accurate attention
            # (The patched forward won't intercept prefill.)
            if self._fp16_keys is None:
                self._fp16_keys = key_states
            else:
                self._fp16_keys = torch.cat(
                    [self._fp16_keys, key_states], dim=-2,
                )
            return self._fp16_keys, self._values
        else:
            # Decode: fused attention uses compressed keys directly.
            # Release FP16 keys on first decode — they are never used
            # by the patched forward (TQ-6c VRAM optimization).
            if not self._prefill_done:
                self._prefill_done = True
                self._fp16_keys = None
            # Return current key_states as dummy (ignored by patched forward)
            return key_states, self._values

    @property
    def compressed_keys(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (indices, norms) for fused attention."""
        return self._key_indices, self._key_norms

    def get_seq_length(self) -> int:
        return self._seq_length

    def get_max_cache_shape(self) -> int:
        return -1

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        kv_offset = 0
        # _seq_length already includes the current query (incremented in update())
        kv_length = self._seq_length
        return kv_length, kv_offset

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            max_length = self._seq_length - abs(max_length)
        if self._seq_length <= max_length:
            return
        self._key_indices = self._key_indices[..., :max_length, :]
        self._key_norms = self._key_norms[..., :max_length]
        if self._fp16_keys is not None:
            self._fp16_keys = self._fp16_keys[..., :max_length, :]
        self._values = self._values[..., :max_length, :]
        self._seq_length = max_length

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self._seq_length == 0:
            return
        self._key_indices = self._key_indices.repeat_interleave(repeats, dim=0)
        self._key_norms = self._key_norms.repeat_interleave(repeats, dim=0)
        if self._fp16_keys is not None:
            self._fp16_keys = self._fp16_keys.repeat_interleave(repeats, dim=0)
        self._values = self._values.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self._seq_length == 0:
            return
        self._key_indices = self._key_indices[indices, ...]
        self._key_norms = self._key_norms[indices, ...]
        if self._fp16_keys is not None:
            self._fp16_keys = self._fp16_keys[indices, ...]
        self._values = self._values[indices, ...]


class TurboQuantCache(DynamicCache):
    """DynamicCache with TurboQuant key compression."""

    def __init__(self, compressor: TurboQuantCompressor):
        self._compressor = compressor
        super().__init__()
        self.layer_class_to_replicate = None
        self.layers = []

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        while len(self.layers) <= layer_idx:
            self.layers.append(TurboQuantLayer(self._compressor))

        return self.layers[layer_idx].update(
            key_states, value_states, cache_kwargs,
        )

    def get_tq_layer(self, layer_idx: int) -> TurboQuantLayer:
        return self.layers[layer_idx]


def patch_model_for_turboquant(model, compressor: TurboQuantCompressor):
    """Monkey-patch Qwen3 attention layers for TurboQuant decode.

    During decode (seq_len=1 query), replaces standard attention with
    fused compressed-key attention. During prefill, uses normal attention.

    Args:
        model: Qwen3 model instance.
        compressor: TurboQuantCompressor with codebook and rotation matrix.

    Returns:
        TurboQuantCache instance to pass to generate().
    """
    from shared_state.turboquant_triton import turboquant_decode_attention

    # Prepare compressor data on model device
    device = next(model.parameters()).device
    compressor._ensure_matrices(device)
    compressor.codebook._ensure_torch(device)
    rot = torch.from_numpy(compressor._rotation_np).float().to(device)
    centroids = compressor.codebook._centroids_torch

    for layer in model.model.layers:
        attn = layer.self_attn
        original_forward = attn.forward
        layer_idx = attn.layer_idx

        def make_patched_forward(orig_fwd, l_idx, attn_module):
            def patched_forward(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values=None,
                cache_position=None,
                **kwargs,
            ):
                # Check if this is decode (seq_len=1) with our cache
                is_decode = (
                    hidden_states.shape[-2] == 1
                    and isinstance(past_key_values, TurboQuantCache)
                )

                if not is_decode:
                    # Prefill: use original attention
                    return orig_fwd(
                        hidden_states, position_embeddings, attention_mask,
                        past_key_values=past_key_values,
                        cache_position=cache_position, **kwargs,
                    )

                # === Decode with fused compressed-key attention ===
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, attn_module.head_dim)

                # Compute Q, K, V + RoPE (same as original)
                query_states = attn_module.q_norm(
                    attn_module.q_proj(hidden_states).view(hidden_shape),
                ).transpose(1, 2)
                key_states = attn_module.k_norm(
                    attn_module.k_proj(hidden_states).view(hidden_shape),
                ).transpose(1, 2)
                value_states = attn_module.v_proj(hidden_states).view(
                    hidden_shape,
                ).transpose(1, 2)

                cos, sin = position_embeddings
                from transformers.models.qwen3.modeling_qwen3 import (
                    apply_rotary_pos_emb,
                )
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin,
                )

                # Update cache (compresses new key, stores value)
                _, value_cache = past_key_values.update(
                    key_states, value_states, l_idx,
                    {"sin": sin, "cos": cos, "cache_position": cache_position},
                )

                # Get compressed key data
                tq_layer = past_key_values.get_tq_layer(l_idx)
                k_indices, k_norms = tq_layer.compressed_keys

                # Fused attention with compressed keys
                attn_output = turboquant_decode_attention(
                    query_states, k_indices, k_norms,
                    centroids, rot, value_cache,
                )

                attn_output = attn_output.to(hidden_states.dtype)
                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                attn_output = attn_module.o_proj(attn_output)
                return attn_output, None

            return patched_forward

        attn.forward = make_patched_forward(original_forward, layer_idx, attn)
