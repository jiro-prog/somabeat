"""Profile TurboQuant decode attention vs rest of model.

Measures per-token GPU time breakdown:
  - fused_attn: turboquant_decode_attention (codebook lookup + matmul)
  - cache_update: key compression + value concat
  - qkv_rope: Q/K/V projection + RoPE (inside patched forward)
  - o_proj: output projection (inside patched forward)
  - other: FFN, LayerNorm, residual, embedding, etc.

Usage:
  .venv/bin/python scripts/profile_turboquant.py
"""

from __future__ import annotations

import logging
import time
import sys
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, ".")
from shared_state.turboquant import TurboQuantCompressor
from llamarcute_live.kv_cache import TurboQuantCache, TurboQuantLayer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Global timing accumulators ──
_timings: dict[str, list[float]] = defaultdict(list)
_token_count = 0


def _cuda_sync_ms(start_event, end_event) -> float:
    """Return elapsed ms between two CUDA events."""
    end_event.synchronize()
    return start_event.elapsed_time(end_event)


def patch_model_for_turboquant_profiled(model, compressor: TurboQuantCompressor):
    """Monkey-patch with per-component CUDA event timing."""
    from shared_state.turboquant_triton import turboquant_decode_attention

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
                is_decode = (
                    hidden_states.shape[-2] == 1
                    and isinstance(past_key_values, TurboQuantCache)
                )

                if not is_decode:
                    return orig_fwd(
                        hidden_states, position_embeddings, attention_mask,
                        past_key_values=past_key_values,
                        cache_position=cache_position, **kwargs,
                    )

                # ── QKV + RoPE ──
                e_start = torch.cuda.Event(enable_timing=True)
                e_qkv = torch.cuda.Event(enable_timing=True)
                e_cache = torch.cuda.Event(enable_timing=True)
                e_attn = torch.cuda.Event(enable_timing=True)
                e_oproj = torch.cuda.Event(enable_timing=True)

                e_start.record()

                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, attn_module.head_dim)

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

                e_qkv.record()

                # ── Cache update (compression) ──
                _, value_cache = past_key_values.update(
                    key_states, value_states, l_idx,
                    {"sin": sin, "cos": cos, "cache_position": cache_position},
                )

                tq_layer = past_key_values.get_tq_layer(l_idx)
                k_indices, k_norms = tq_layer.compressed_keys

                e_cache.record()

                # ── Fused attention ──
                attn_output = turboquant_decode_attention(
                    query_states, k_indices, k_norms,
                    centroids, rot, value_cache,
                )

                e_attn.record()

                # ── Output projection ──
                attn_output = attn_output.to(hidden_states.dtype)
                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                attn_output = attn_module.o_proj(attn_output)

                e_oproj.record()

                # Record timings (only for layer 0 to avoid noise)
                if l_idx == 0:
                    torch.cuda.synchronize()
                    _timings["qkv_rope"].append(e_start.elapsed_time(e_qkv))
                    _timings["cache_update"].append(e_qkv.elapsed_time(e_cache))
                    _timings["fused_attn"].append(e_cache.elapsed_time(e_attn))
                    _timings["o_proj"].append(e_cache.elapsed_time(e_oproj) - e_cache.elapsed_time(e_attn))

                return attn_output, None

            return patched_forward

        attn.forward = make_patched_forward(original_forward, layer_idx, attn)


def main():
    model_name = "Qwen/Qwen3-8B"
    max_new_tokens = 64  # Short run for profiling
    prompt = "日本の四季について短く説明してください。"

    logger.info("=== TurboQuant Speed Profiler ===")
    logger.info("Loading model...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    device = next(model.parameters()).device
    head_dim = model.config.head_dim
    logger.info("Model loaded. head_dim=%d, device=%s", head_dim, device)
    logger.info(
        "VRAM: allocated=%.0fMiB, reserved=%.0fMiB",
        torch.cuda.memory_allocated() / 1024**2,
        torch.cuda.memory_reserved() / 1024**2,
    )

    # TurboQuant setup
    compressor = TurboQuantCompressor(head_dim=head_dim, bits=3)
    patch_model_for_turboquant_profiled(model, compressor)
    cache = TurboQuantCache(compressor)

    # Tokenize
    input_text = f"{prompt}\n/no_think"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    n_input = input_ids.shape[1]
    logger.info("Input tokens: %d", n_input)

    # Warmup (1 token)
    logger.info("Warmup...")
    with torch.no_grad():
        warmup_cache = TurboQuantCache(compressor)
        _ = model.generate(
            input_ids=input_ids,
            max_new_tokens=2,
            do_sample=False,
            past_key_values=warmup_cache,
            pad_token_id=tokenizer.pad_token_id,
        )
    del warmup_cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _timings.clear()
    logger.info(
        "Post-warmup VRAM: allocated=%.0fMiB",
        torch.cuda.memory_allocated() / 1024**2,
    )

    # Profiled run
    logger.info("Profiling %d tokens...", max_new_tokens)
    cache = TurboQuantCache(compressor)

    torch.cuda.synchronize()
    t_start = time.monotonic()

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            past_key_values=cache,
            pad_token_id=tokenizer.pad_token_id,
        )

    torch.cuda.synchronize()
    t_total = time.monotonic() - t_start

    n_output = output_ids.shape[1] - n_input
    text = tokenizer.decode(output_ids[0][n_input:], skip_special_tokens=True)
    logger.info("Output tokens: %d", n_output)
    logger.info("Output: %s", text[:200])
    logger.info("")

    # ── Results ──
    logger.info("=== TIMING BREAKDOWN (per decode token, layer 0) ===")
    n_layers = len(model.model.layers)

    for key in ["qkv_rope", "cache_update", "fused_attn", "o_proj"]:
        vals = _timings.get(key, [])
        if vals:
            avg = sum(vals) / len(vals)
            # Extrapolate to all layers
            total_per_token = avg * n_layers
            logger.info(
                "  %-15s  layer0=%.3fms  x%d layers=%.1fms/token",
                key, avg, n_layers, total_per_token,
            )

    # Estimate "other" (FFN, layernorm, etc.)
    attn_total_per_token = sum(
        sum(v) / len(v) * n_layers
        for v in _timings.values()
        if v
    )
    total_per_token_ms = (t_total / n_output) * 1000 if n_output > 0 else 0
    other_per_token = total_per_token_ms - attn_total_per_token

    logger.info("")
    logger.info("  %-15s  %.1fms/token", "ATTN TOTAL", attn_total_per_token)
    logger.info("  %-15s  %.1fms/token", "OTHER (FFN etc)", other_per_token)
    logger.info("  %-15s  %.1fms/token", "TOTAL", total_per_token_ms)
    logger.info("")
    logger.info("  Wall time: %.1fs for %d tokens = %.2f tok/s", t_total, n_output, n_output / t_total if t_total > 0 else 0)
    logger.info(
        "  Peak VRAM (decode only): %.0fMiB",
        torch.cuda.max_memory_allocated() / 1024**2,
    )
    logger.info(
        "  Current VRAM: allocated=%.0fMiB, reserved=%.0fMiB",
        torch.cuda.memory_allocated() / 1024**2,
        torch.cuda.memory_reserved() / 1024**2,
    )


if __name__ == "__main__":
    main()
