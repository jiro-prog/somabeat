"""Tests for model separation between llamarcute-live and SleepyJean.

Invariant: llamarcute-live uses a base model (no fine-tuning),
SleepyJean's LoRA pipeline updates only the 'sleepyjean' ollama model.
The two must never point to the same model when fine-tuning is active.

Added after 2026-03-22 incident where fine-tuning collapse in sleepyjean
model caused llamarcute-live dialogue quality to degrade.
"""

import yaml
from pathlib import Path

import pytest


CONFIG_PATH = Path(__file__).parent.parent / "config" / "system.yaml"


@pytest.fixture
def config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestModelSeparation:
    """Ensure llamarcute-live and sleepyjean use different models."""

    def test_config_has_separate_model_keys(self, config):
        """Both sections must declare their own ollama_model."""
        ll_model = config["llamarcute_live"].get("ollama_model")
        sj_model = config["sleepyjean"].get("ollama_model")
        assert ll_model is not None, "llamarcute_live.ollama_model missing from config"
        assert sj_model is not None, "sleepyjean.ollama_model missing from config"

    def test_models_are_different(self, config):
        """llamarcute-live and sleepyjean must not use the same model name."""
        ll_model = config["llamarcute_live"]["ollama_model"]
        sj_model = config["sleepyjean"]["ollama_model"]
        assert ll_model != sj_model, (
            f"Model separation violated: both use '{ll_model}'. "
            "llamarcute-live must use a base model, sleepyjean uses fine-tuned."
        )

    def test_llamarcute_model_is_base(self, config):
        """llamarcute-live should use a base (non-fine-tuned) model."""
        ll_model = config["llamarcute_live"]["ollama_model"]
        # Fine-tuned models are named 'sleepyjean' or contain 'fine' / 'lora'
        assert "sleepyjean" not in ll_model.lower(), (
            f"llamarcute-live model '{ll_model}' appears to be a fine-tuned model"
        )

    def test_sleepyjean_model_is_dedicated(self, config):
        """sleepyjean should use its own dedicated model name."""
        sj_model = config["sleepyjean"]["ollama_model"]
        assert sj_model == "sleepyjean", (
            f"sleepyjean model should be 'sleepyjean', got '{sj_model}'"
        )

    def test_lora_pipeline_targets_sleepyjean_only(self, config):
        """lora_train.py must only update the 'sleepyjean' ollama model."""
        sj_root = config.get("sleepyjean", {}).get("root_path", "")
        lora_train_path = Path(sj_root) / "scripts" / "night" / "lora_train.py"
        if not lora_train_path.exists():
            pytest.skip(f"lora_train.py not found at {lora_train_path}")

        content = lora_train_path.read_text(encoding="utf-8")

        # _reload_ollama() must create model named "sleepyjean"
        assert '"sleepyjean"' in content or "'sleepyjean'" in content, (
            "lora_train.py must target the 'sleepyjean' ollama model"
        )

        # Must NOT reference qwen3:8b (the llamarcute-live model)
        assert "qwen3:8b" not in content, (
            "lora_train.py must not reference the llamarcute-live model (qwen3:8b)"
        )

    def test_default_model_in_ollama_client(self):
        """ollama_client DEFAULT_MODEL must match config."""
        from llamarcute_live.ollama_client import DEFAULT_MODEL
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        ll_model = config["llamarcute_live"]["ollama_model"]
        assert DEFAULT_MODEL == ll_model, (
            f"ollama_client DEFAULT_MODEL '{DEFAULT_MODEL}' != "
            f"config llamarcute_live.ollama_model '{ll_model}'"
        )
