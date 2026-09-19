#!/usr/bin/env python3
"""Test: 2x consecutive load→unload to verify VRAM release.

Success criterion: VRAM allocated < 16 MiB after each unload.
"""

import sys
sys.path.insert(0, ".")

import torch
import yaml

THRESHOLD_MIB = 50  # generous threshold; expect ~16 MiB


def vram_mib():
    return torch.cuda.memory_allocated() / 1024**2


def main():
    with open("config/system.yaml") as f:
        config = yaml.safe_load(f)

    llm_cfg = config["llamarcute_live"]["llm"]

    from llamarcute_live.llm_inference import FieldAwareLLM
    llm = FieldAwareLLM(
        quantization=llm_cfg["quantization"],
        gptq_model=llm_cfg.get("gptq_model", "AlphaGaO/Qwen3-8B-GPTQ"),
        model_name=llm_cfg.get("transformers_model", "Qwen/Qwen3-8B"),
        kv_cache_bits=llm_cfg.get("kv_cache_bits", 0),
    )

    baseline = vram_mib()
    print(f"Baseline VRAM: {baseline:.0f} MiB")

    results = []
    for i in range(1, 3):
        print(f"\n=== Round {i}: load ===")
        llm.load()
        after_load = vram_mib()
        print(f"  After load:   {after_load:.0f} MiB")

        print(f"=== Round {i}: unload ===")
        llm.unload()
        after_unload = vram_mib()
        print(f"  After unload: {after_unload:.0f} MiB")

        passed = after_unload < THRESHOLD_MIB
        results.append((i, after_load, after_unload, passed))
        print(f"  {'PASS' if passed else 'FAIL'} (threshold: <{THRESHOLD_MIB} MiB)")

    print("\n=== Summary ===")
    print(f"| Round | After load (MiB) | After unload (MiB) | Result |")
    print(f"|-------|------------------|--------------------|--------|")
    for rnd, al, au, ok in results:
        print(f"| {rnd}     | {al:.0f}              | {au:.0f}                | {'PASS' if ok else 'FAIL'}   |")

    all_pass = all(r[3] for r in results)
    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
