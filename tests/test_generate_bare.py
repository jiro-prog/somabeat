"""Tests for FieldAwareLLM.generate_bare() — input_ids-based inference without field injection.

Tests:
1. generate_bare returns a response
2. system_prompt is reflected (different prompts → different tokenization)
3. generate_with_field and generate_bare can alternate without KV cache interference
4. Post-processing (<think> removal) works
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest
import torch


def _make_mock_llm():
    """Create a FieldAwareLLM with mocked model and tokenizer."""
    from llamarcute_live.llm_inference import FieldAwareLLM

    llm = FieldAwareLLM.__new__(FieldAwareLLM)
    llm._model_name = "mock"
    llm._device = "cpu"
    llm._max_new_tokens = 64
    llm._temperature = 0.7
    llm._kv_cache_bits = 0
    llm._quantization = "gptq_marlin"
    llm._gptq_model = "mock"
    llm._compressor = None
    llm._gptq_wrapper = None
    llm._model_patched = False
    llm._base_text_norm = None

    # Mock tokenizer
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 0

    # apply_chat_template returns dict with input_ids
    def mock_apply_chat_template(messages, add_generation_prompt=False, return_tensors=None):
        # Return different lengths for different system prompts to test prompt reflection
        total_len = 10
        for m in messages:
            total_len += len(m["content"]) // 10
        return {"input_ids": torch.ones(1, total_len, dtype=torch.long)}

    tokenizer.apply_chat_template = mock_apply_chat_template

    llm._tokenizer = tokenizer

    # Mock model
    model = MagicMock()
    model.parameters.side_effect = lambda: iter([torch.zeros(1)])
    model.model.embed_tokens = MagicMock(
        side_effect=lambda ids: torch.randn(ids.shape[0], ids.shape[1], 32),
    )

    llm._model = model

    return llm


class TestGenerateBare:
    def test_returns_response(self):
        """generate_bare returns a non-None response string."""
        llm = _make_mock_llm()
        # Mock generate to return token ids
        llm._model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        llm._tokenizer.decode.return_value = "Hello world"

        response, duration = asyncio.get_event_loop().run_until_complete(
            llm.generate_bare(system_prompt="Test", user_input="Hi")
        )
        assert response is not None
        assert isinstance(response, str)
        assert duration > 0

    def test_system_prompt_reflected(self):
        """Different system prompts produce different input_ids lengths."""
        llm = _make_mock_llm()
        llm._model.generate.return_value = torch.tensor([[1, 2, 3]])
        llm._tokenizer.decode.return_value = "response"

        captured_kwargs = []

        def capture_generate(**kwargs):
            captured_kwargs.append(kwargs)
            return torch.tensor([[1, 2, 3]])

        llm._model.generate.side_effect = capture_generate

        loop = asyncio.get_event_loop()

        # Short prompt
        loop.run_until_complete(
            llm.generate_bare(system_prompt="Be brief.", user_input="Hi")
        )
        # Long prompt
        loop.run_until_complete(
            llm.generate_bare(
                system_prompt="You are a very detailed assistant that explains everything thoroughly.",
                user_input="Hi",
            )
        )

        assert len(captured_kwargs) == 2
        short_len = captured_kwargs[0]["input_ids"].shape[1]
        long_len = captured_kwargs[1]["input_ids"].shape[1]
        assert long_len > short_len, (
            f"Longer system prompt should produce more tokens: {short_len} vs {long_len}"
        )

    def test_alternate_with_generate_with_field(self):
        """generate_with_field and generate_bare can alternate without error."""
        import numpy as np

        llm = _make_mock_llm()
        llm._model.generate.return_value = torch.tensor([[1, 2, 3]])
        llm._tokenizer.decode.return_value = "response"

        loop = asyncio.get_event_loop()

        # generate_with_field (no field embeddings — text only via inputs_embeds)
        r1, _ = loop.run_until_complete(
            llm.generate_with_field(
                system_prompt="Test", user_input="Hello",
                field_embeddings=None,
            )
        )
        assert r1 is not None

        # generate_bare (input_ids path)
        r2, _ = loop.run_until_complete(
            llm.generate_bare(system_prompt="Test", user_input="Hello")
        )
        assert r2 is not None

        # Back to generate_with_field
        r3, _ = loop.run_until_complete(
            llm.generate_with_field(
                system_prompt="Test", user_input="Hello",
                field_embeddings=None,
            )
        )
        assert r3 is not None

    def test_postprocess_strips_think_tags(self):
        """Post-processing removes <think> blocks from output."""
        llm = _make_mock_llm()
        llm._model.generate.return_value = torch.tensor([[1, 2, 3]])
        llm._tokenizer.decode.return_value = (
            "<think>internal reasoning here</think>The actual response"
        )

        response, _ = asyncio.get_event_loop().run_until_complete(
            llm.generate_bare(system_prompt="Test", user_input="Hi")
        )
        assert "think" not in response.lower()
        assert "internal reasoning" not in response
        assert "actual response" in response

    def test_no_think_appended_by_default(self):
        """By default, /no_think is appended to system content."""
        llm = _make_mock_llm()
        llm._model.generate.return_value = torch.tensor([[1, 2, 3]])
        llm._tokenizer.decode.return_value = "response"

        captured_messages = []
        original_apply = llm._tokenizer.apply_chat_template

        def capture_apply(messages, **kwargs):
            captured_messages.append(messages)
            return original_apply(messages, **kwargs)

        llm._tokenizer.apply_chat_template = capture_apply

        asyncio.get_event_loop().run_until_complete(
            llm.generate_bare(system_prompt="Be helpful.", user_input="Hi")
        )

        # Check that /no_think was appended to system content
        assert len(captured_messages) == 1
        sys_msg = captured_messages[0][0]
        assert sys_msg["role"] == "system"
        assert sys_msg["content"].endswith("/no_think")

    def test_no_think_can_be_disabled(self):
        """When no_think=False, /no_think is not appended."""
        llm = _make_mock_llm()
        llm._model.generate.return_value = torch.tensor([[1, 2, 3]])
        llm._tokenizer.decode.return_value = "response"

        captured_messages = []
        original_apply = llm._tokenizer.apply_chat_template

        def capture_apply(messages, **kwargs):
            captured_messages.append(messages)
            return original_apply(messages, **kwargs)

        llm._tokenizer.apply_chat_template = capture_apply

        asyncio.get_event_loop().run_until_complete(
            llm.generate_bare(
                system_prompt="Be helpful.", user_input="Hi", no_think=False,
            )
        )

        sys_msg = captured_messages[0][0]
        assert not sys_msg["content"].endswith("/no_think")
