"""Stage F claim-level grounding check.

Stage E (in :mod:`src.agent.v1_answer_verifier`) requires the answer's
novel proper-noun entities to co-occur with at least one question anchor
*somewhere* in the cited cells. That's necessary but not sufficient: the
generator can still splice signal A from cell 1 with signal B from cell 2
and produce a sentence that names both — Stage E passes because each side
co-occurs with the question anchor in *some* cell, but no single cell
actually supports the conjunction the sentence asserts.

Stage F closes that gap. For every atomic claim with at least one
distinctive signal (numeric, month name, or proper-noun run), it
requires the signals to appear in the SAME cited cell. The first cell
that satisfies all of a claim's signals is the supporting cell;
absence of any such cell rejects the whole answer.

Claims with zero distinctive signals (e.g. a generic stitching clause
"and this was the case for many years") are skipped — there is nothing
to verify. The deliberate scope of Stage F is named-thing co-occurrence,
not paraphrase fidelity; Stage A still owns coverage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from src.agent.claim_decomposer import ClaimSpan, decompose
from src.agent.v1_answer_verifier import (
    _extract_capitalised_runs,
    _extract_numerics,
    _normalise_entity,
    _normalise_for_anchor,
)


@dataclass
class ClaimVerification:
    """Per-claim Stage F outcome.

    ``ok`` is True iff the claim had no distinctive signals (skipped)
    OR at least one cited cell contains all of the claim's normalised
    signals. ``supporting_cell_id`` is the first cell that satisfied
    the conjunction; None when skipped or rejected. ``missing_in_best``
    is the residual set of signals that the best partial-match cell
    still didn't contain — useful for diagnosing why a cross-cell
    splice was rejected.
    """

    claim_text: str
    start: int
    end: int
    numerics: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    skipped: bool = False
    ok: bool = False
    supporting_cell_id: int | None = None
    missing_in_best: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "claim_text": self.claim_text,
            "start": self.start,
            "end": self.end,
            "numerics": self.numerics,
            "entities": self.entities,
            "skipped": self.skipped,
            "ok": self.ok,
            "supporting_cell_id": self.supporting_cell_id,
            "missing_in_best": self.missing_in_best,
        }


@dataclass
class ClaimVerifierReport:
    """Whole-answer Stage F outcome."""

    grounded: bool
    n_claims: int
    n_skipped: int
    n_verified: int
    n_rejected: int
    failed: list[ClaimVerification] = field(default_factory=list)
    per_claim: list[ClaimVerification] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "grounded": self.grounded,
            "n_claims": self.n_claims,
            "n_skipped": self.n_skipped,
            "n_verified": self.n_verified,
            "n_rejected": self.n_rejected,
            "failed": [f.as_dict() for f in self.failed],
            "per_claim": [c.as_dict() for c in self.per_claim],
        }


def _claim_signals(
    claim_text: str,
    *,
    question_numerics: set[str],
    question_anchors_norm: set[str],
) -> tuple[list[str], list[str]]:
    """Return ``(numerics, entity_norms)`` for ``claim_text``.

    Numerics from the question are excluded (they don't prove
    grounding). Entity runs that normalise to a substring of any
    question anchor are excluded for the same reason.
    """
    numerics = [n for n in _extract_numerics(claim_text)
                if n not in question_numerics]

    entity_norms: list[str] = []
    seen: set[str] = set()
    for span in _extract_capitalised_runs(claim_text):
        norm = _normalise_entity(span)
        if not norm or norm in seen:
            continue
        # Entity normalises to (or contains) a question anchor → not a
        # novel claim signal.
        if any(norm == qa or norm in qa or qa in norm
               for qa in question_anchors_norm):
            continue
        seen.add(norm)
        entity_norms.append(norm)
    return numerics, entity_norms


def _cell_satisfies(
    cell_text_norm: str, numerics: Sequence[str], entities: Sequence[str]
) -> tuple[bool, list[str]]:
    """Return ``(all_present, missing)``. ``cell_text_norm`` is
    already normalised via :func:`_normalise_for_anchor` (lowercased,
    punctuation → spaces, whitespace collapsed). Numerics are matched
    after the same normalisation so a comma-separated form like
    ``658,000`` in the answer matches ``658 000`` in the normalised
    cell text."""
    missing: list[str] = []
    for n in numerics:
        if _normalise_for_anchor(n) not in cell_text_norm:
            missing.append(n)
    for e in entities:
        if e not in cell_text_norm:
            missing.append(e)
    return (not missing), missing


def verify_claims(
    answer: str,
    cells: Sequence[tuple[int, str]],
    *,
    question: str = "",
) -> ClaimVerifierReport:
    """Stage F: every claim's distinctive signals must co-occur in some
    single cited cell.

    ``cells`` is a sequence of ``(cell_id, cell_text)`` pairs in the
    order they were presented to the generator (and therefore the order
    citations point at).
    """
    claims = decompose(answer)

    question_numerics = set(_extract_numerics(question)) if question else set()
    question_anchors_norm = set()
    if question:
        for span in _extract_capitalised_runs(question):
            norm = _normalise_entity(span)
            if norm:
                question_anchors_norm.add(norm)

    cell_blobs_norm: list[tuple[int, str]] = [
        (cid, _normalise_for_anchor(text)) for cid, text in cells
    ]

    per_claim: list[ClaimVerification] = []
    failed: list[ClaimVerification] = []
    n_skipped = 0
    n_verified = 0
    n_rejected = 0

    for span in claims:
        numerics, entities = _claim_signals(
            span.text,
            question_numerics=question_numerics,
            question_anchors_norm=question_anchors_norm,
        )
        if not numerics and not entities:
            cv = ClaimVerification(
                claim_text=span.text,
                start=span.start,
                end=span.end,
                skipped=True,
                ok=True,
            )
            n_skipped += 1
            per_claim.append(cv)
            continue

        best_missing: list[str] | None = None
        supporter: int | None = None
        for cid, blob in cell_blobs_norm:
            ok, missing = _cell_satisfies(blob, numerics, entities)
            if ok:
                supporter = cid
                best_missing = []
                break
            if best_missing is None or len(missing) < len(best_missing):
                best_missing = missing

        cv = ClaimVerification(
            claim_text=span.text,
            start=span.start,
            end=span.end,
            numerics=numerics,
            entities=entities,
            ok=supporter is not None,
            supporting_cell_id=supporter,
            missing_in_best=best_missing or [],
        )
        per_claim.append(cv)
        if cv.ok:
            n_verified += 1
        else:
            n_rejected += 1
            failed.append(cv)

    grounded = not failed
    return ClaimVerifierReport(
        grounded=grounded,
        n_claims=len(claims),
        n_skipped=n_skipped,
        n_verified=n_verified,
        n_rejected=n_rejected,
        failed=failed,
        per_claim=per_claim,
    )


__all__ = ["ClaimVerification", "ClaimVerifierReport", "verify_claims"]
