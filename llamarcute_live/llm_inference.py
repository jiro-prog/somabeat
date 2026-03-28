"""LLM inference with field embedding injection via transformers.

Replaces Ollama for cognitive functions that read from the shared field.
Uses Qwen3-8B with 4-bit quantization and inputs_embeds injection.

Supported quantization backends:
  - gptq_marlin: GPTQ + Marlin kernel (W4A16 fused GEMV). Default.
  - nf4: bitsandbytes NF4 (legacy, slower on bandwidth-limited GPUs).

Injection layout:
  [system_prompt_tokens] [field_signal_embeds] [user_input_tokens]
                          ↑ FieldReceptor output injected here

Reference: instructions_fieldreceptor_integration.md T6
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class FieldAwareLLM:
    """LLM inference engine with field embedding injection capability.

    Manages model lifecycle and provides two inference modes:
    - generate_with_field(): injects FieldReceptor output into input sequence
    - generate_text_only(): pure text inference (no field signals)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        device: str = "cuda",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        kv_cache_bits: int = 0,
        quantization: str = "gptq_marlin",
        gptq_model: str = "AlphaGaO/Qwen3-8B-GPTQ",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._kv_cache_bits = kv_cache_bits
        self._quantization = quantization
        self._gptq_model = gptq_model
        self._model = None
        self._tokenizer = None
        self._compressor = None
        self._model_patched = False
        self._base_text_norm: float | None = None  # calibrated on first inference

    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the model with 4-bit quantization."""
        if self._model is not None:
            return

        torch.cuda.empty_cache()

        if self._quantization == "gptq_marlin":
            self._load_gptq_marlin()
        else:
            self._load_nf4()

        self._model.eval()
        logger.info("Model loaded. Device: %s", next(self._model.parameters()).device)

        # Initialize TurboQuant compressor for KV cache compression
        if self._kv_cache_bits > 0:
            from shared_state.turboquant import TurboQuantCompressor
            head_dim = self._model.config.head_dim
            self._compressor = TurboQuantCompressor(
                head_dim=head_dim, bits=self._kv_cache_bits,
            )
            logger.info(
                "TurboQuant KV cache: %d-bit, head_dim=%d",
                self._kv_cache_bits, head_dim,
            )

        logger.info(
            "VRAM after load: allocated=%.0fMiB, reserved=%.0fMiB",
            torch.cuda.memory_allocated() / 1024**2,
            torch.cuda.memory_reserved() / 1024**2,
        )

    def _load_gptq_marlin(self) -> None:
        """Load GPTQ model with Marlin kernel (W4A16 fused GEMV)."""
        from gptqmodel import GPTQModel

        logger.info("Loading %s with GPTQ+Marlin...", self._gptq_model)
        wrapper = GPTQModel.load(
            self._gptq_model, device_map="auto", backend="marlin",
        )
        self._model = wrapper.model

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._gptq_model, trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

    def _load_nf4(self) -> None:
        """Load model with bitsandbytes NF4 quantization (legacy)."""
        from transformers import BitsAndBytesConfig

        logger.info("Loading %s with NF4 quantization...", self._model_name)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

    def unload(self) -> None:
        """Free GPU memory."""
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            self._model_patched = False
            if self._compressor is not None:
                del self._compressor
                self._compressor = None
            torch.cuda.empty_cache()
            logger.info("Model unloaded, GPU memory freed.")

    @property
    def _embed_layer(self):
        return self._model.model.embed_tokens

    @property
    def _model_device(self):
        return next(self._model.parameters()).device

    async def generate_with_field(
        self,
        system_prompt: str,
        user_input: str,
        field_embeddings: NDArray[np.float32] | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> tuple[str | None, float]:
        """Generate a response with field embeddings injected.

        Args:
            system_prompt: System prompt text.
            user_input: User message text.
            field_embeddings: (K, 4096) array from FieldReceptor.transduce().
                              None or empty → text-only inference.
            max_new_tokens: Override default max tokens.
            temperature: Override default temperature.

        Returns:
            (response_text, duration_seconds). response_text is None on error.
        """
        start = time.monotonic()

        try:
            self.load()

            import re

            device = self._model_device
            embed_layer = self._embed_layer
            max_tok = max_new_tokens or self._max_new_tokens
            temp = temperature or self._temperature

            # Build chat-formatted token sequence via apply_chat_template.
            # /no_think is included in system content so it lands inside
            # <|im_start|>system\n{content}\n/no_think<|im_end|>
            #
            # Field embeddings are injected between system and user turns.
            # We split the template output at the boundary, embed each part,
            # then concatenate with field embeddings in between.

            sys_content = system_prompt + "\n/no_think"
            messages_sys = [{"role": "system", "content": sys_content}]
            messages_user = [{"role": "user", "content": user_input}]

            # System turn tokens (with template formatting)
            sys_ids = self._tokenizer.apply_chat_template(
                messages_sys,
                add_generation_prompt=False,
                return_tensors="pt",
            )["input_ids"].to(device)

            # User turn + generation prompt tokens
            user_ids = self._tokenizer.apply_chat_template(
                messages_user,
                add_generation_prompt=True,
                return_tensors="pt",
            )["input_ids"].to(device)

            # Get text embeddings
            with torch.no_grad():
                sys_embeds = embed_layer(sys_ids)    # (1, T_sys, H)
                user_embeds = embed_layer(user_ids)  # (1, T_user, H)

            # Build combined input — dtype must match embed_layer output
            if field_embeddings is not None and len(field_embeddings) > 0:
                field_tensor = torch.from_numpy(field_embeddings).to(
                    device=device, dtype=sys_embeds.dtype,
                )
                if field_tensor.dim() == 2:
                    field_tensor = field_tensor.unsqueeze(0)  # (1, K, H)

                # Scale field embeddings to match text token norm range.
                # FieldReceptor output norms (~22) are ~14x larger than text
                # token norms (~1.6), which distorts attention scores.
                # Uniform scaling preserves relative norm ratios (signal
                # concentration) per shared_field_design.md 3.1.
                #
                # base_text_norm is calibrated once on first inference and
                # fixed thereafter. This ensures the scaling factor is
                # independent of sequence length (conversation history).
                # See llamarcute_live_design.md 3.4.
                if self._base_text_norm is None:
                    self._base_text_norm = sys_embeds.norm(dim=-1).mean().item()
                    logger.info(
                        "Calibrated base_text_norm=%.4f (from %d sys tokens)",
                        self._base_text_norm, sys_embeds.shape[1],
                    )

                text_norm_current = sys_embeds.norm(dim=-1).mean()
                field_mean_norm = field_tensor.norm(dim=-1).mean().clamp(min=1e-8)
                field_norms_pre = field_tensor.squeeze(0).norm(dim=-1)
                scale = self._base_text_norm / field_mean_norm
                field_tensor = field_tensor * scale
                field_norms_post = field_tensor.squeeze(0).norm(dim=-1)
                logger.info(
                    "Field embedding norms: pre=[min=%.2f, mean=%.2f, max=%.2f] "
                    "post=[min=%.2f, mean=%.2f, max=%.2f] "
                    "scale=%.4f, base_norm=%.2f, current_sys_norm=%.2f, ratio_post=%.2f",
                    field_norms_pre.min(), field_norms_pre.mean(), field_norms_pre.max(),
                    field_norms_post.min(), field_norms_post.mean(), field_norms_post.max(),
                    scale, self._base_text_norm, text_norm_current,
                    field_norms_post.mean() / self._base_text_norm,
                )

                combined = torch.cat([sys_embeds, field_tensor, user_embeds], dim=1)
            else:
                combined = torch.cat([sys_embeds, user_embeds], dim=1)

            # Attention mask (all ones — no padding in single-sequence inference)
            attention_mask = torch.ones(
                1, combined.shape[1], dtype=torch.long, device=device,
            )

            # Generate with TurboQuant KV cache compression
            generate_kwargs = dict(
                inputs_embeds=combined,
                attention_mask=attention_mask,
                max_new_tokens=max_tok,
                do_sample=temp > 0,
                temperature=temp if temp > 0 else 1.0,
                top_p=0.9,
                repetition_penalty=1.3,
                pad_token_id=self._tokenizer.pad_token_id,
            )
            if self._compressor is not None:
                from llamarcute_live.kv_cache import (
                    TurboQuantCache,
                    patch_model_for_turboquant,
                )
                if not self._model_patched:
                    patch_model_for_turboquant(self._model, self._compressor)
                    self._model_patched = True
                cache = TurboQuantCache(self._compressor)
                if field_embeddings is not None and len(field_embeddings) > 0:
                    cache.enable_field_pruning(
                        field_start=sys_ids.shape[1],
                        field_count=field_embeddings.shape[0],
                    )
                generate_kwargs["past_key_values"] = cache

            with torch.no_grad():
                output_ids = self._model.generate(**generate_kwargs)

            # Decode with skip_special_tokens=False to preserve <think> tags,
            # then strip thinking blocks, then remove remaining special tokens.
            raw = self._tokenizer.decode(
                output_ids[0], skip_special_tokens=False,
            )
            # Strip closed and unclosed thinking blocks
            response = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
            response = re.sub(r"<think>.*", "", response, flags=re.DOTALL)
            # Strip Qwen3 special tokens (<|im_start|>, <|im_end|>, <|endoftext|>, etc.)
            response = re.sub(r"<\|[^|]*\|>", "", response)
            response = response.strip()

            duration = time.monotonic() - start
            n_field = field_embeddings.shape[0] if field_embeddings is not None else 0
            n_input = combined.shape[1]
            n_output = output_ids.shape[1]
            tps = n_output / duration if duration > 0 else 0
            logger.info(
                "FieldAwareLLM: %d chars, %.1fs, %.1f tok/s "
                "(input=%d [sys=%d + field=%d + user=%d], output=%d)",
                len(response), duration, tps,
                n_input, sys_ids.shape[1], n_field, user_ids.shape[1],
                n_output,
            )
            return response, duration

        except Exception as e:
            logger.error("FieldAwareLLM error: %s", e)
            return None, time.monotonic() - start

    async def generate_text_only(
        self,
        system_prompt: str,
        user_input: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> tuple[str | None, float]:
        """Text-only generation (no field injection). Convenience wrapper."""
        return await self.generate_with_field(
            system_prompt=system_prompt,
            user_input=user_input,
            field_embeddings=None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
