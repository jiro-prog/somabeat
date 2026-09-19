"""Tests for model configuration of llamarcute-live.

Ensures llamarcute-live uses a base model (not a fine-tuned model).
The old SleepyJean model separation tests are removed — new SleepyJean
is non-LLM and does not use Ollama.
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
    """Ensure llamarcute-live uses a base model via FieldAwareLLM."""

    def test_gptq_model_is_configured(self, config):
        """GPTQ model path should be set in config."""
        llm_cfg = config["llamarcute_live"]["llm"]
        assert "gptq_model" in llm_cfg
        assert "GPTQ" in llm_cfg["gptq_model"]

    def test_quantization_backend_valid(self, config):
        """Quantization backend should be gptq_marlin or nf4."""
        llm_cfg = config["llamarcute_live"]["llm"]
        assert llm_cfg["quantization"] in ("gptq_marlin", "nf4")
