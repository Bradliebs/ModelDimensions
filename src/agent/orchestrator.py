"""Stateful memory bank and orchestration pipeline.

``MemoryBank`` is a thin, stateful wrapper around the *frozen* concept-cell
construction (``concept_cells.geometry.build_concept_cells``). It does not
reimplement any geometry. Its only own behaviour is a fixed-radius
normalisation of incoming vectors so the one-shot threshold construction is
always valid (see ``MemoryBank`` docstring).

``Orchestrator`` is the v0.8 pipeline: user text -> intent -> (encode) ->
bank -> response policy.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from concept_cells.geometry import build_concept_cells
from concept_cells.persistence import (
    MemoryRecord,
    read_records,
    write_records,
)
from slm.controller import SLMController
from slm.schemas import (
    GroundedResponse,
    IntentType,
    MemoryQueryResult,
    SLMControllerDecision,
)

from .response_policy import build_response


# ---------- Encoder interface ----------

@runtime_checkable
class EncoderProtocol(Protocol):
    """Anything that can turn a single string into a 1-D float vector.

    The bank only ever needs per-text encoding, so the interface is one
    method. This keeps the deterministic offline encoder and a real MiniLM
    encoder interchangeable.
    """

    def encode_one(self, text: str) -> np.ndarray:
        ...


class DeterministicEncoder:
    """Seeded, content-addressed encoder for fully offline runs.

    The vector for a given text is a deterministic function of that text's
    hash, so writing then querying the *same* text reproduces the *same*
    vector exactly (which the control encoders in ``concept_cells.encoders``
    do not guarantee, since they are seeded by item index). This is what lets
    tests and exp06 run without any model download.

    Vectors are drawn iid Gaussian then scaled to sit inside the unit ball;
    the bank re-normalises to its own fixed radius on write/query anyway.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

    def encode_one(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big", signed=False)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.dim).astype(np.float64)
        n = np.linalg.norm(v)
        if n < 1e-12:
            v = np.ones(self.dim) / np.sqrt(self.dim)
            n = 1.0
        # unit vector * 0.5 keeps it strictly inside the unit ball before the
        # bank's own radius normalisation.
        return (v / n * 0.5).astype(np.float32)


# ---------- Memory bank ----------

class MemoryBank:
    """A stateful set of concept-cell memories built on the frozen core.

    Design constraints honoured here:

    * Geometry is delegated to ``build_concept_cells`` — never reimplemented.
    * Every stored vector is L2-normalised to a fixed radius ``r`` (< 1) so the
      ``norm_minus_eps`` threshold ``theta = r - epsilon`` is always valid and
      a cell is guaranteed to fire for its own item (``<w_i, x_i> = r > theta``).
      Cross-firing then requires cosine similarity above ``r / (r) ...`` i.e.
      near-duplicates only; this is the bank's own preprocessing, not a change
      to the core construction.
    * Firing stays strictly per-cell: a query fires cell ``i`` iff
      ``<w_i, q> > theta_i``. "Silence" means no cell fired.
    """

    def __init__(self, encoder: EncoderProtocol, epsilon: float = 0.05,
                 radius: float = 0.9):
        if not 0.0 < radius < 1.0:
            raise ValueError("radius must be in (0, 1)")
        if not 0.0 <= epsilon < radius:
            raise ValueError("epsilon must satisfy 0 <= epsilon < radius")
        self.encoder = encoder
        self.epsilon = float(epsilon)
        self.radius = float(radius)
        self._records: List[MemoryRecord] = []
        self._next_id = 1
        # Lazily rebuilt cell arrays (w, theta) cached after each mutation.
        self._w: Optional[np.ndarray] = None
        self._theta: Optional[np.ndarray] = None

    # -- internal helpers --

    def _normalise(self, v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        if n < 1e-12:
            raise ValueError("cannot store a zero vector")
        return (v / n * self.radius).astype(np.float32)

    def _rebuild_cells(self) -> None:
        if not self._records:
            self._w = None
            self._theta = None
            return
        emb = np.array([r.vector for r in self._records], dtype=np.float32)
        # Frozen core construction. Vectors already sit at radius r, so theta
        # = r - epsilon for every cell.
        self._w, self._theta = build_concept_cells(
            emb, epsilon=self.epsilon, threshold_scheme="norm_minus_eps"
        )

    def _mint_id(self) -> str:
        mid = f"mem-{self._next_id:04d}"
        self._next_id += 1
        return mid

    # -- mutating operations --

    def write(self, canonical_text: str, source: str = "user",
              tags: Optional[Sequence[str]] = None) -> MemoryRecord:
        """Encode and store one memory; returns the created record."""
        vec = self._normalise(self.encoder.encode_one(canonical_text))
        rec = MemoryRecord(
            memory_id=self._mint_id(),
            canonical_text=canonical_text,
            vector=vec.tolist(),
            threshold=self.radius - self.epsilon,
            source=source,
            tags=list(tags) if tags else [],
        )
        self._records.append(rec)
        self._rebuild_cells()
        return rec

    def delete(self, memory_id: str) -> bool:
        """Remove a memory by exact id. Returns True if something was removed."""
        before = len(self._records)
        self._records = [r for r in self._records if r.memory_id != memory_id]
        removed = len(self._records) != before
        if removed:
            self._rebuild_cells()
        return removed

    def bind(self, memory_ids: Sequence[str], bound_group_id: str) -> List[str]:
        """Tag existing memories with a shared binding group id.

        Binding is recorded at the metadata level: it groups memories without
        collapsing their individual cells, so each bound memory remains
        independently retrievable and auditable. Returns the ids that were
        actually updated (ids not present are skipped).
        """
        updated: List[str] = []
        idset = set(memory_ids)
        for rec in self._records:
            if rec.memory_id in idset:
                rec.bound_group_id = bound_group_id
                updated.append(rec.memory_id)
        return updated

    # -- read operations --

    def query(self, query_text: str) -> MemoryQueryResult:
        """Query the bank; returns fired memory ids and per-fire margins."""
        if not self._records or self._w is None or self._theta is None:
            return MemoryQueryResult(query_text=query_text, silent=True)
        q = self._normalise(self.encoder.encode_one(query_text))
        activations = self._w @ q  # (n,)
        margins = activations - self._theta  # (n,)
        fired_mask = margins > 0.0
        fired_ids = [
            self._records[i].memory_id
            for i in range(len(self._records))
            if fired_mask[i]
        ]
        fired_margins = [float(margins[i]) for i in range(len(self._records))
                         if fired_mask[i]]
        return MemoryQueryResult(
            query_text=query_text,
            fired_memory_ids=fired_ids,
            margins=fired_margins,
            silent=len(fired_ids) == 0,
        )

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        for rec in self._records:
            if rec.memory_id == memory_id:
                return rec
        return None

    def records(self) -> List[MemoryRecord]:
        return list(self._records)

    def as_items(self) -> List[dict]:
        """Lightweight view for the controller's binding suggestion."""
        return [
            {"memory_id": r.memory_id, "canonical_text": r.canonical_text}
            for r in self._records
        ]

    # -- persistence --

    def save(self, path: str) -> None:
        write_records(path, self._records)

    def load(self, path: str) -> None:
        self._records = read_records(path)
        # Re-derive the id counter so newly written ids do not collide.
        max_n = 0
        for r in self._records:
            try:
                max_n = max(max_n, int(r.memory_id.split("-")[-1]))
            except (ValueError, IndexError):
                continue
        self._next_id = max_n + 1
        self._rebuild_cells()


# ---------- Orchestrator ----------

class Orchestrator:
    """End-to-end pipeline: user text -> intent -> bank -> grounded response.

    The SLM controller is optional. With ``controller=None`` the orchestrator
    treats every input as a query against the bank, so the memory system runs
    with no language model in the loop at all.
    """

    def __init__(self, bank: MemoryBank,
                 controller: Optional[SLMController] = None):
        self.bank = bank
        self.controller = controller

    def handle(self, user_text: str) -> GroundedResponse:
        if self.controller is None:
            # No SLM: pure retrieval.
            result = self.bank.query(user_text)
            return build_response(result, self.bank)

        decision: SLMControllerDecision = self.controller.classify_intent(
            user_text
        )
        intent = decision.intent.intent

        if intent == IntentType.WRITE and decision.write_candidate is not None:
            rec = self.bank.write(
                decision.write_candidate.canonical_text,
                source=decision.write_candidate.source,
                tags=decision.write_candidate.tags,
            )
            return GroundedResponse(
                text=f"Stored memory {rec.memory_id}.",
                cited_memory_ids=[rec.memory_id],
                memory_used=True,
                refused=False,
            )

        if intent == IntentType.DELETE:
            mem_id = decision.delete_memory_id
            if not mem_id:
                return GroundedResponse(
                    text="Refused: a delete needs an explicit memory id.",
                    refused=True,
                )
            removed = self.bank.delete(mem_id)
            if removed:
                return GroundedResponse(
                    text=f"Deleted memory {mem_id}.",
                    cited_memory_ids=[mem_id],
                    memory_used=True,
                )
            return GroundedResponse(
                text=f"Refused: no memory with id {mem_id}.",
                refused=True,
            )

        if intent == IntentType.BIND:
            candidate = self.controller.suggest_bindings(self.bank.as_items())
            if candidate is None:
                return GroundedResponse(
                    text="Refused: need at least two memories to bind.",
                    refused=True,
                )
            updated = self.bank.bind(
                candidate.memory_ids, candidate.bound_group_id
            )
            if not updated:
                return GroundedResponse(
                    text="Refused: none of the proposed memories exist.",
                    refused=True,
                )
            return GroundedResponse(
                text=(
                    f"Bound {len(updated)} memories into "
                    f"{candidate.bound_group_id}."
                ),
                cited_memory_ids=updated,
                memory_used=True,
            )

        if intent == IntentType.QUERY:
            query_text = decision.query_text or user_text
            result = self.bank.query(query_text)
            return build_response(result, self.bank)

        # UNKNOWN intent: refuse rather than guess.
        return GroundedResponse(
            text="Refused: could not classify this request as a memory action.",
            refused=True,
        )
