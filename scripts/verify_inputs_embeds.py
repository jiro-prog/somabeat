"""T6-a: Verify that generate(inputs_embeds=...) works with Qwen3-8B + 4-bit.

Tests three scenarios:
  1. Normal generate (input_ids) — baseline
  2. generate(inputs_embeds=...) from text — same text, embedding route
  3. generate(inputs_embeds=...) with injected vectors — field signal simulation

Prerequisites: Ollama stopped (VRAM needed).

Usage:
  python scripts/verify_inputs_embeds.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_model():
    model_name = "Qwen/Qwen3-8B"
    print(f"Loading {model_name} with 4-bit quantization...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded. Device: {next(model.parameters()).device}")
    return model, tokenizer


def get_embedding_layer(model):
    """Get the token embedding layer from the model."""
    return model.model.embed_tokens


def test_normal_generate(model, tokenizer):
    """Test 1: Normal generation with input_ids."""
    print("\n--- Test 1: Normal generate (input_ids) ---")
    prompt = "日本の首都は"
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            temperature=1.0,
        )

    response = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"  Prompt: {prompt}")
    print(f"  Response: {response}")
    return True


def test_inputs_embeds_from_text(model, tokenizer):
    """Test 2: Generate using inputs_embeds derived from same text."""
    print("\n--- Test 2: generate(inputs_embeds=...) from text ---")
    prompt = "日本の首都は"
    device = next(model.parameters()).device
    embed_layer = get_embedding_layer(model)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    # Get embeddings manually
    with torch.no_grad():
        embeds = embed_layer(input_ids)  # (1, T, 4096)

    # Generate using inputs_embeds instead of input_ids
    with torch.no_grad():
        output_ids = model.generate(
            inputs_embeds=embeds,
            max_new_tokens=30,
            do_sample=False,
            temperature=1.0,
        )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"  Prompt: {prompt}")
    print(f"  Response: {response}")
    return True


def test_inputs_embeds_with_injection(model, tokenizer):
    """Test 3: Inject synthetic vectors between system prompt and user input."""
    print("\n--- Test 3: generate(inputs_embeds=...) with injection ---")
    device = next(model.parameters()).device
    embed_layer = get_embedding_layer(model)
    hidden_size = model.config.hidden_size

    system_text = "あなたは日本語で答えるAIです。"
    user_text = "東京タワーについて教えて。"

    sys_ids = tokenizer(system_text, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    user_ids = tokenizer(user_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    with torch.no_grad():
        sys_embeds = embed_layer(sys_ids)    # (1, T_sys, H)
        user_embeds = embed_layer(user_ids)  # (1, T_user, H)

    # Simulate field signal injection: 3 random vectors scaled to match embedding norms
    embed_norm = sys_embeds.norm(dim=-1).mean().item()
    injected = torch.randn(1, 3, hidden_size, device=device, dtype=sys_embeds.dtype)
    injected = injected * embed_norm / injected.norm(dim=-1, keepdim=True)

    # Concatenate: [system] [injected] [user]
    combined = torch.cat([sys_embeds, injected, user_embeds], dim=1)
    print(f"  sys_tokens={sys_embeds.shape[1]}, injected={injected.shape[1]}, user_tokens={user_embeds.shape[1]}")
    print(f"  combined: {combined.shape}")

    with torch.no_grad():
        output_ids = model.generate(
            inputs_embeds=combined,
            max_new_tokens=50,
            do_sample=False,
            temperature=1.0,
        )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"  Response: {response[:200]}")
    return True


def main():
    model, tokenizer = load_model()

    results = {}
    for name, test_fn in [
        ("normal_generate", test_normal_generate),
        ("inputs_embeds_text", test_inputs_embeds_from_text),
        ("inputs_embeds_injection", test_inputs_embeds_with_injection),
    ]:
        try:
            test_fn(model, tokenizer)
            results[name] = "PASS"
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[name] = f"FAIL: {e}"

    print(f"\n{'='*60}")
    print("VERIFICATION RESULTS")
    print(f"{'='*60}")
    for name, status in results.items():
        print(f"  {name}: {status}")
    print(f"{'='*60}")

    all_pass = all(v == "PASS" for v in results.values())
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
