"""Shared Field interface definitions.

Defines the core abstractions for inter-system communication:
Signal, FieldReading, SharedField Protocol, FieldEncoder Protocol, FieldObserver Protocol.

Reference: llamarcute_live_design.md section 7.2
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SignalOrigin:
    """Identifies the source system and context of a signal."""
    system: str       # e.g. "llamarcute_live", "sleepyjean", "immune"
    context: str      # e.g. "dialogue", "difficulty", "knowledge_update"


@dataclass(frozen=True)
class Signal:
    """A unit of information emitted into the shared field.

    Immutable. The embedding is NOT normalized — its L2 norm carries
    "concentration" semantics.  No text or extra metadata fields exist;
    the field carries embeddings only (design principle 5.3).
    """
    signal_id: str
    embedding: NDArray[np.float32]   # non-normalized vector
    emitted_at: datetime
    origin: SignalOrigin

    @staticmethod
    def create(
        embedding: NDArray[np.float32],
        origin: SignalOrigin,
    ) -> Signal:
        return Signal(
            signal_id=uuid.uuid4().hex,
            embedding=embedding,
            emitted_at=datetime.now(),
            origin=origin,
        )


@dataclass(frozen=True)
class WeightedSignal:
    """A signal observed through sense(), with relevance and decay applied."""
    signal: Signal
    relevance: float       # cosine similarity to query (0..1)
    decay_factor: float    # temporal decay (0..1)
    effective_weight: float  # relevance * decay_factor * ||embedding||


@dataclass(frozen=True)
class FieldReading:
    """Result of a sense() operation."""
    signals: list[WeightedSignal]
    observed_at: datetime
    query_embedding: NDArray[np.float32]


@dataclass
class PerceiveParams:
    """Parameters controlling how perceive() reads the field."""
    decay_fn: DecayFunction | None = None  # None → default ExponentialDecay
    min_strength: float = 0.1
    max_signals: int = 50
    time_horizon: timedelta | None = None  # None → no time cutoff
    strength_exponent: float = 0.5  # Weber-Fechner: strength = decay × norm^α


@dataclass(frozen=True)
class PerceivedSignal:
    """A signal observed through perceive(), with decay and strength applied."""
    signal: Signal
    decay_factor: float
    strength: float       # decay_factor × ‖embedding‖


@dataclass(frozen=True)
class FieldPerception:
    """Result of a perceive() operation."""
    signals: list[PerceivedSignal]   # strength降順
    perceived_at: datetime


@dataclass
class SenseParams:
    """Parameters controlling how sense() reads the field."""
    decay_fn: DecayFunction | None = None  # None → default ExponentialDecay
    max_signals: int = 10
    min_relevance: float = 0.1
    time_horizon: timedelta | None = None  # None → no time cutoff


# ---------------------------------------------------------------------------
# Decay functions
# ---------------------------------------------------------------------------

@runtime_checkable
class DecayFunction(Protocol):
    """Protocol for temporal decay functions."""
    def __call__(self, elapsed: timedelta) -> float:
        """Return decay factor in [0, 1] for the given elapsed time."""
        ...


@dataclass(frozen=True)
class ExponentialDecay:
    """Exponential decay: exp(-lambda * hours).

    half_life_hours controls how fast signals fade.
    """
    half_life_hours: float = 24.0

    def __call__(self, elapsed: timedelta) -> float:
        hours = elapsed.total_seconds() / 3600.0
        if hours < 0:
            return 1.0
        lam = math.log(2) / self.half_life_hours
        return math.exp(-lam * hours)


# ---------------------------------------------------------------------------
# Purge / Snapshot
# ---------------------------------------------------------------------------

@dataclass
class PurgeCriteria:
    """Criteria for removing signals from the field.

    Origin-based filtering is intentionally absent — the non-directionality
    principle (design principle 2) applies to purge as well as perceive.
    """
    older_than: timedelta | None = None
    max_signals_to_purge: int | None = None


@dataclass(frozen=True)
class PurgeResult:
    """Result of a purge() operation."""
    purged_count: int
    remaining_count: int


@dataclass(frozen=True)
class FieldSnapshot:
    """A point-in-time snapshot of all signals in the field."""
    signals: list[Signal]
    taken_at: datetime
    total_count: int


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

@runtime_checkable
class SharedField(Protocol):
    """The shared field — the medium for indirect coordination between systems.

    Design note — sense() vs snapshot() usage:

        sense() returns signals ranked by cosine similarity to a query vector.
        This is appropriate for semantic retrieval: "find information related
        to X" (e.g. dialogue context, self-awareness queries).

        snapshot() returns all signals unranked. Use snapshot() + context
        filter when the goal is to retrieve signals of a specific origin.context
        reliably (e.g. "difficulty", "rotation_task", "immune"). Cosine
        similarity ranking is unsuitable here because unrelated signals with
        higher norms or incidental textual overlap can crowd out the target
        signals from the top-K results.

        The field intentionally has no origin-based query API (design principle
        2: non-directionality). When catalog-style retrieval by context is
        needed, snapshot() + in-memory filter is the correct pattern.
    """

    async def emit(self, signal: Signal) -> None:
        """Fire-and-forget signal emission into the field."""
        ...

    async def perceive(
        self,
        params: PerceiveParams | None = None,
    ) -> FieldPerception:
        """Perceive the field — retrieve all signals with decay-based strength.

        Unlike sense(), perceive() does not rank by cosine similarity to a query.
        It returns all signals above the strength threshold, sorted by strength.
        """
        ...

    async def sense(
        self,
        query_embedding: NDArray[np.float32],
        params: SenseParams | None = None,
    ) -> FieldReading:
        """Sense the field with a query vector. Returns relevant signals.

        Use for semantic retrieval where cosine similarity ranking is desired.
        Do NOT use when the goal is to retrieve all signals of a specific
        origin.context — use snapshot() + context filter instead.
        """
        ...

    async def purge(self, criteria: PurgeCriteria) -> PurgeResult:
        """Remove signals matching the criteria."""
        ...

    async def snapshot(self) -> FieldSnapshot:
        """Take a snapshot of the current field state.

        Use with context filter for catalog-style retrieval of specific
        signal types (e.g. difficulty, rotation_task, immune).
        """
        ...


@runtime_checkable
class FieldEncoder(Protocol):
    """Encodes text into embeddings for the shared field."""

    def encode(self, text: str) -> NDArray[np.float32]:
        """Encode text to a non-normalized embedding vector."""
        ...

    def encode_image(self, image: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Encode an image to a non-normalized embedding vector in the shared space."""
        ...

    @property
    def dimensionality(self) -> int:
        """Return the dimensionality of the embedding space."""
        ...


@runtime_checkable
class FieldReceptor(Protocol):
    """Transforms field embeddings into agent-native representations.

    Maps from field embedding space to the agent's internal representation space
    (e.g. LLM input embedding space).
    """

    def transduce(
        self,
        field_embeddings: list[NDArray[np.float32]],
        strengths: list[float],
    ) -> NDArray[np.float32]:
        """Transform field embeddings into agent-native representation.

        Args:
            field_embeddings: List of field-space embedding vectors.
            strengths: Corresponding strength values for scaling.

        Returns:
            Transformed embeddings in agent-native space.
        """
        ...

    def field_dimensionality(self) -> int:
        """Return the dimensionality of the field embedding space."""
        ...

    def agent_dimensionality(self) -> int:
        """Return the dimensionality of the agent's internal space."""
        ...


@runtime_checkable
class FieldObserver(Protocol):
    """Observer for field operations — used for debugging and monitoring."""

    def on_emit(self, signal: Signal) -> None:
        ...

    def on_perceive(self, perception: FieldPerception) -> None:
        ...

    def on_purge(self, result: PurgeResult) -> None:
        ...
