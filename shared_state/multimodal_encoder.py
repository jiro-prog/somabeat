"""MultimodalFieldEncoder — unified text+image encoder for the shared field.

Uses e5-small for text and SigLIP ViT-B for images, with learned projection
heads (from Phase B) to map both modalities into the shared D=384 space.

All models run on CPU by default to avoid VRAM contention with Ollama.

Reference: docs/design/instructions_vision_phase_c.md C-1b
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoProcessor

from shared_state.projection_heads import ProjectionImg, ProjectionText

logger = logging.getLogger(__name__)

class MultimodalFieldEncoder:
    """FieldEncoder that supports both text and image encoding.

    Text: e5-small -> ProjectionText -> D=384
    Image: SigLIP ViT-B -> ProjectionImg -> D=384

    Both modalities share the same vector space thanks to the learned
    projection heads trained in Phase B.
    """

    def __init__(
        self,
        e5_model_name: str = "intfloat/multilingual-e5-small",
        siglip_model_name: str = "google/siglip2-base-patch16-256",
        projection_text_path: str = "data/vision_phase_b/alpha_0.5/projection_text.pt",
        projection_img_path: str = "data/vision_phase_b/alpha_0.5/projection_img.pt",
        device: str = "cpu",
    ) -> None:
        self._device = device
        self._dimensionality = 384

        # Text encoder (e5-small)
        logger.info("Loading text encoder: %s", e5_model_name)
        self._text_encoder = SentenceTransformer(e5_model_name)

        # Image encoder (SigLIP ViT-B vision tower) — CPU to avoid VRAM contention
        logger.info("Loading image encoder: %s (device=%s)", siglip_model_name, device)
        full_model = AutoModel.from_pretrained(siglip_model_name).to(device)
        self._vision_model = full_model.vision_model
        self._image_processor = AutoProcessor.from_pretrained(siglip_model_name)
        self._vision_model.eval()

        # Learned projection heads
        self._projection_text = self._load_projection(
            ProjectionText, projection_text_path,
        )
        self._projection_img = self._load_projection(
            ProjectionImg, projection_img_path,
        )

        logger.info("MultimodalFieldEncoder ready (D=%d, device=%s)", self._dimensionality, device)

    @staticmethod
    def _load_projection(cls: type[nn.Module], path: str) -> nn.Module:
        model = cls()
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model

    # --- Text encoding ---

    def encode(self, text: str) -> NDArray[np.float32]:
        """Encode text to a non-normalized embedding vector (passage prefix)."""
        return self.encode_for_emit(text)

    def encode_for_emit(self, text: str) -> NDArray[np.float32]:
        """Encode text for emission (passage prefix) through projection."""
        with torch.no_grad():
            raw = self._text_encoder.encode(
                f"passage: {text}", normalize_embeddings=False,
            )
            raw_tensor = torch.tensor(raw, dtype=torch.float32).unsqueeze(0)
            projected = self._projection_text(raw_tensor)
        return projected.squeeze(0).numpy()

    def encode_for_sense(self, text: str) -> NDArray[np.float32]:
        """Encode text for sensing/querying (query prefix) through projection."""
        with torch.no_grad():
            raw = self._text_encoder.encode(
                f"query: {text}", normalize_embeddings=False,
            )
            raw_tensor = torch.tensor(raw, dtype=torch.float32).unsqueeze(0)
            projected = self._projection_text(raw_tensor)
        return projected.squeeze(0).numpy()

    # --- Image encoding ---

    def encode_image(self, image: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Encode a single image to D-dim shared space vector (non-normalized)."""
        with torch.no_grad():
            inputs = self._image_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            outputs = self._vision_model(pixel_values=inputs["pixel_values"])
            img_features = outputs.pooler_output  # (1, 768)
            projected = self._projection_img(img_features)  # (1, 384)
        return projected.squeeze(0).numpy()

    # --- Properties ---

    @property
    def dimensionality(self) -> int:
        return self._dimensionality
