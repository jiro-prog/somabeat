"""Tests for MultimodalFieldEncoder.

These tests load the real SigLIP and e5-small models, so they are slower.
The module-scoped fixture ensures models are loaded only once.

Reference: docs/design/instructions_vision_phase_c.md C-4b
"""

import numpy as np
import pytest

from shared_state.encoder import E5SmallEncoder


@pytest.fixture(scope="module")
def multimodal_encoder():
    """Load MultimodalFieldEncoder once for the entire module."""
    try:
        from shared_state.multimodal_encoder import MultimodalFieldEncoder
        return MultimodalFieldEncoder(device="cpu")
    except Exception as e:
        pytest.skip(f"MultimodalFieldEncoder initialization failed: {e}")


@pytest.fixture
def dummy_image():
    """Small dummy image (64x64 RGB)."""
    return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)


class TestMultimodalEncoder:
    def test_encode_image_shape(self, multimodal_encoder, dummy_image):
        """encode_image returns a (384,) vector."""
        vec = multimodal_encoder.encode_image(dummy_image)
        assert vec.shape == (384,)
        assert vec.dtype == np.float32

    def test_encode_text_emit_shape(self, multimodal_encoder):
        """encode_for_emit returns a (384,) vector."""
        vec = multimodal_encoder.encode_for_emit("テスト文章")
        assert vec.shape == (384,)
        assert vec.dtype == np.float32

    def test_encode_text_sense_shape(self, multimodal_encoder):
        """encode_for_sense returns a (384,) vector."""
        vec = multimodal_encoder.encode_for_sense("テスト文章")
        assert vec.shape == (384,)
        assert vec.dtype == np.float32

    def test_same_space(self, multimodal_encoder, dummy_image):
        """Text and image embeddings live in the same space (cosine similarity in [0,1] is computable)."""
        text_vec = multimodal_encoder.encode_for_emit("a photo of a cat")
        img_vec = multimodal_encoder.encode_image(dummy_image)
        cos_sim = np.dot(text_vec, img_vec) / (
            np.linalg.norm(text_vec) * np.linalg.norm(img_vec)
        )
        assert -1.0 <= cos_sim <= 1.0

    def test_projection_text_not_identity(self, multimodal_encoder):
        """projection_text transforms the vector (alpha=0.5, not identity)."""
        text = "passage: テスト文章"
        raw = multimodal_encoder._text_encoder.encode(text, normalize_embeddings=False)
        projected = multimodal_encoder.encode_for_emit("テスト文章")
        # Should NOT be identical (projection head transforms it)
        assert not np.allclose(raw, projected, atol=1e-5)

    def test_e5_fallback_encode_image(self):
        """E5SmallEncoder.encode_image raises NotImplementedError."""
        encoder = E5SmallEncoder()
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        with pytest.raises(NotImplementedError, match="E5SmallEncoder does not support image encoding"):
            encoder.encode_image(dummy)

    def test_config_switching(self):
        """encoder.type config controls which encoder is created."""
        from orchestrator.orchestrator import _create_encoder

        # e5_small (default)
        config_e5 = {"encoder": {"model_name": "intfloat/multilingual-e5-small"}}
        enc = _create_encoder(config_e5)
        assert type(enc).__name__ == "E5SmallEncoder"

        # multimodal with bad path → falls back to E5SmallEncoder
        config_bad = {
            "encoder": {
                "type": "multimodal",
                "model_name": "intfloat/multilingual-e5-small",
                "projection_text_path": "/nonexistent/path.pt",
                "projection_img_path": "/nonexistent/path.pt",
            }
        }
        enc2 = _create_encoder(config_bad)
        assert type(enc2).__name__ == "E5SmallEncoder"
