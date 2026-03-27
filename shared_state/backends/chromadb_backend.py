"""ChromaDB implementation of the SharedField protocol.

Reference: llamarcute_live_design.md section 7.3, phase1_taskflow.md T3
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta

import warnings

import chromadb
import numpy as np
from numpy.typing import NDArray

from shared_state.interface import (
    ExponentialDecay,
    FieldObserver,
    FieldPerception,
    FieldReading,
    FieldSnapshot,
    PerceiveParams,
    PerceivedSignal,
    PurgeCriteria,
    PurgeResult,
    SenseParams,
    Signal,
    SignalOrigin,
    WeightedSignal,
)

logger = logging.getLogger(__name__)

_ISO_FMT = "%Y-%m-%dT%H:%M:%S.%f"


class ChromaDBField:
    """SharedField backed by ChromaDB.

    ChromaDB normalizes embeddings by default, so we store the original L2 norm
    in metadata and reconstruct effective_weight at sense() time.
    """

    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str = "shared_field",
        observer: FieldObserver | None = None,
    ) -> None:
        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.EphemeralClient()

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._observer = observer

    async def emit(self, signal: Signal) -> None:
        norm = float(np.linalg.norm(signal.embedding))
        metadata = {
            "origin_system": signal.origin.system,
            "origin_context": signal.origin.context,
            "emitted_at": signal.emitted_at.strftime(_ISO_FMT),
            "norm": norm,
        }
        if signal.extra:
            metadata["extra_json"] = json.dumps(signal.extra, ensure_ascii=False)
        self._collection.add(
            ids=[signal.signal_id],
            embeddings=[signal.embedding.tolist()],
            metadatas=[metadata],
            documents=[""],
        )
        if self._observer:
            self._observer.on_emit(signal)

    async def perceive(
        self,
        params: PerceiveParams | None = None,
    ) -> FieldPerception:
        if params is None:
            params = PerceiveParams()

        decay_fn = params.decay_fn or ExponentialDecay()
        now = datetime.now()

        total = self._collection.count()
        if total == 0:
            return FieldPerception(signals=[], perceived_at=now)

        all_data = self._collection.get(include=["embeddings", "metadatas"])
        perceived: list[PerceivedSignal] = []

        for sid, emb_raw, meta in zip(
            all_data["ids"], all_data["embeddings"], all_data["metadatas"]
        ):
            emitted_at = datetime.strptime(meta["emitted_at"], _ISO_FMT)

            if params.time_horizon and (now - emitted_at) > params.time_horizon:
                continue

            elapsed = now - emitted_at
            decay_factor = decay_fn(elapsed)
            norm = float(meta["norm"])
            strength = decay_factor * norm

            if strength < params.min_strength:
                continue

            extra = {}
            if "extra_json" in meta:
                try:
                    extra = json.loads(meta["extra_json"])
                except (json.JSONDecodeError, TypeError):
                    pass

            signal = Signal(
                signal_id=sid,
                embedding=np.array(emb_raw, dtype=np.float32),
                emitted_at=emitted_at,
                origin=SignalOrigin(
                    system=meta["origin_system"],
                    context=meta["origin_context"],
                ),
                extra=extra,
            )
            perceived.append(PerceivedSignal(
                signal=signal,
                decay_factor=decay_factor,
                strength=strength,
            ))

        perceived.sort(key=lambda p: p.strength, reverse=True)
        perceived = perceived[: params.max_signals]

        perception = FieldPerception(signals=perceived, perceived_at=now)
        if self._observer:
            self._observer.on_perceive(perception)
        return perception

    async def sense(
        self,
        query_embedding: NDArray[np.float32],
        params: SenseParams | None = None,
    ) -> FieldReading:
        warnings.warn(
            "sense() is deprecated. Use perceive() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if params is None:
            params = SenseParams()

        decay_fn = params.decay_fn or ExponentialDecay()
        now = datetime.now()

        total = self._collection.count()
        if total == 0:
            reading = FieldReading(
                signals=[], observed_at=now, query_embedding=query_embedding,
            )
            return reading

        n_results = min(total, params.max_signals * 3)
        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            include=["embeddings", "metadatas", "distances"],
        )

        weighted: list[WeightedSignal] = []
        ids = results["ids"][0]
        embeddings = results["embeddings"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for i, sid in enumerate(ids):
            meta = metadatas[i]
            emitted_at = datetime.strptime(meta["emitted_at"], _ISO_FMT)

            if params.time_horizon and (now - emitted_at) > params.time_horizon:
                continue

            # cosine distance → similarity
            relevance = max(0.0, 1.0 - distances[i])
            if relevance < params.min_relevance:
                continue

            elapsed = now - emitted_at
            decay_factor = decay_fn(elapsed)
            norm = float(meta["norm"])
            effective_weight = relevance * decay_factor * norm

            extra = {}
            if "extra_json" in meta:
                try:
                    extra = json.loads(meta["extra_json"])
                except (json.JSONDecodeError, TypeError):
                    pass
            signal = Signal(
                signal_id=sid,
                embedding=np.array(embeddings[i], dtype=np.float32),
                emitted_at=emitted_at,
                origin=SignalOrigin(
                    system=meta["origin_system"],
                    context=meta["origin_context"],
                ),
                extra=extra,
            )
            weighted.append(WeightedSignal(
                signal=signal,
                relevance=relevance,
                decay_factor=decay_factor,
                effective_weight=effective_weight,
            ))

        weighted.sort(key=lambda w: w.effective_weight, reverse=True)
        weighted = weighted[: params.max_signals]

        reading = FieldReading(
            signals=weighted, observed_at=now, query_embedding=query_embedding,
        )
        if self._observer:
            self._observer.on_sense(reading, query_text=None)
        return reading

    async def purge(self, criteria: PurgeCriteria) -> PurgeResult:
        before_count = self._collection.count()
        if before_count == 0:
            result = PurgeResult(purged_count=0, remaining_count=0)
            if self._observer:
                self._observer.on_purge(result)
            return result

        all_data = self._collection.get(include=["metadatas"])
        ids_to_delete: list[str] = []
        now = datetime.now()

        for sid, meta in zip(all_data["ids"], all_data["metadatas"]):
            match = True
            emitted_at = datetime.strptime(meta["emitted_at"], _ISO_FMT)

            if criteria.older_than and (now - emitted_at) < criteria.older_than:
                match = False
            if criteria.origin_system and meta["origin_system"] != criteria.origin_system:
                match = False
            if criteria.origin_context and meta["origin_context"] != criteria.origin_context:
                match = False

            if match:
                ids_to_delete.append(sid)
                if criteria.max_signals_to_purge and len(ids_to_delete) >= criteria.max_signals_to_purge:
                    break

        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)

        remaining = self._collection.count()
        result = PurgeResult(
            purged_count=len(ids_to_delete),
            remaining_count=remaining,
        )
        if self._observer:
            self._observer.on_purge(result)
        return result

    async def snapshot(self) -> FieldSnapshot:
        now = datetime.now()
        all_data = self._collection.get(include=["embeddings", "metadatas"])

        signals: list[Signal] = []
        for sid, emb, meta in zip(
            all_data["ids"], all_data["embeddings"], all_data["metadatas"]
        ):
            extra = {}
            if "extra_json" in meta:
                try:
                    extra = json.loads(meta["extra_json"])
                except (json.JSONDecodeError, TypeError):
                    pass
            signals.append(Signal(
                signal_id=sid,
                embedding=np.array(emb, dtype=np.float32),
                emitted_at=datetime.strptime(meta["emitted_at"], _ISO_FMT),
                origin=SignalOrigin(
                    system=meta["origin_system"],
                    context=meta["origin_context"],
                ),
                extra=extra,
            ))

        return FieldSnapshot(
            signals=signals, taken_at=now, total_count=len(signals),
        )
