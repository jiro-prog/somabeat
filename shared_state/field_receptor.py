"""FieldReceptor implementation — transforms field embeddings to LLM input space.

Architecture: 1-hidden MLP (384 → 512 → 4096)
  - Determined by T4 CKA/Procrustes measurement:
    CKA=0.545, Procrustes=0.413 → linear insufficient, 1-hidden MLP required.
  - Target: Qwen3-8B embedding layer output (mean pooling), hidden_size=4096.

Reference: instructions_fieldreceptor_integration.md T5
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray


class FieldReceptorImpl(nn.Module):
    """Transforms field embeddings (384d) to LLM embedding space (4096d).

    Uses a 1-hidden-layer MLP with LayerNorm and GELU activation.
    """

    def __init__(self, field_dim: int = 384, agent_dim: int = 4096, hidden_dim: int = 512):
        super().__init__()
        self._field_dim = field_dim
        self._agent_dim = agent_dim
        self.layers = nn.Sequential(
            nn.Linear(field_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, agent_dim),
        )

    def transduce(
        self,
        field_embeddings: list[NDArray[np.float32]],
        strengths: list[float],
    ) -> NDArray[np.float32]:
        """Transform field embeddings into LLM-native representation.

        Args:
            field_embeddings: List of K field-space vectors, each (field_dim,).
            strengths: Corresponding strength values for scaling.

        Returns:
            (K, agent_dim) array in LLM embedding space.
        """
        if not field_embeddings:
            return np.zeros((0, self._agent_dim), dtype=np.float32)

        # Scale by strength before transformation
        scaled = np.stack([
            emb * s for emb, s in zip(field_embeddings, strengths)
        ], axis=0)  # (K, field_dim)

        x = torch.from_numpy(scaled).float()
        with torch.no_grad():
            out = self.forward(x)
        return out.cpu().numpy()

    def field_dimensionality(self) -> int:
        return self._field_dim

    def agent_dimensionality(self) -> int:
        return self._agent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> FieldReceptorImpl:
        """Load a trained FieldReceptor from a state dict file."""
        state = torch.load(path, map_location=device, weights_only=True)
        field_dim = state["layers.0.weight"].shape[1]
        agent_dim = state["layers.3.weight"].shape[0]
        hidden_dim = state["layers.0.weight"].shape[0]
        model = cls(field_dim=field_dim, agent_dim=agent_dim, hidden_dim=hidden_dim)
        model.load_state_dict(state)
        model.eval()
        return model.to(device)
