"""Tests for SensoryVision module.

Uses mock encoder and field to test thalamic filter logic without
loading real models.

Reference: docs/design/instructions_vision_phase_c.md C-4b
"""

import asyncio

import numpy as np
import pytest

from sensory.vision import SensoryVision
from shared_state.interface import Signal, SignalOrigin


class MockField:
    """Records emitted signals for assertion."""

    def __init__(self):
        self.emitted: list[Signal] = []

    async def emit(self, signal: Signal) -> None:
        self.emitted.append(signal)


class MockImageEncoder:
    """Returns deterministic embeddings based on image content."""
    dimensionality = 384

    def encode_image(self, image):
        seed = int(image.mean() * 1000) % (2**31)
        rng = np.random.RandomState(seed)
        return rng.randn(384).astype(np.float32)

    def encode(self, text):
        return np.random.randn(384).astype(np.float32)

    def encode_for_emit(self, text):
        return self.encode(text)

    def encode_for_sense(self, text):
        return self.encode(text)


@pytest.fixture
def vision_setup():
    encoder = MockImageEncoder()
    field = MockField()
    vision = SensoryVision(encoder=encoder, field=field, thalamus_threshold=0.1)
    return vision, field


class TestSensoryVision:
    def test_first_frame_always_emits(self, vision_setup):
        """The first frame is always emitted (no previous embedding to compare)."""
        vision, field = vision_setup
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = asyncio.get_event_loop().run_until_complete(vision.process_frame(image))
        assert result is True
        assert len(field.emitted) == 1

    def test_similar_frame_suppressed(self, vision_setup):
        """Identical images are suppressed by the thalamic filter."""
        vision, field = vision_setup
        image = np.full((64, 64, 3), 128, dtype=np.uint8)
        asyncio.get_event_loop().run_until_complete(vision.process_frame(image))
        # Same image → same embedding → cosine_dist ≈ 0 → suppressed
        result = asyncio.get_event_loop().run_until_complete(vision.process_frame(image))
        assert result is False
        assert len(field.emitted) == 1

    def test_different_frame_emits(self, vision_setup):
        """Sufficiently different images are both emitted."""
        vision, field = vision_setup
        img1 = np.zeros((64, 64, 3), dtype=np.uint8)
        img2 = np.full((64, 64, 3), 255, dtype=np.uint8)
        asyncio.get_event_loop().run_until_complete(vision.process_frame(img1))
        result = asyncio.get_event_loop().run_until_complete(vision.process_frame(img2))
        assert result is True
        assert len(field.emitted) == 2

    def test_clear_resets_state(self, vision_setup):
        """After clear(), the same image is emitted again (treated as first frame)."""
        vision, field = vision_setup
        image = np.full((64, 64, 3), 100, dtype=np.uint8)
        asyncio.get_event_loop().run_until_complete(vision.process_frame(image))
        vision.clear()
        result = asyncio.get_event_loop().run_until_complete(vision.process_frame(image))
        assert result is True
        assert len(field.emitted) == 2

    def test_emit_signal_format(self, vision_setup):
        """Emitted signals have origin.system='sensory:vision'."""
        vision, field = vision_setup
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        asyncio.get_event_loop().run_until_complete(vision.process_frame(image))
        signal = field.emitted[0]
        assert signal.origin.system == "sensory:vision"
        assert signal.origin.context == "frame"
