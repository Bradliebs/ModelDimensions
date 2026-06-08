"""Acceptance tests for the reading-engine objective.

The roadmap states two numeric pass/fail criteria for the trained model:

* **Ignorance Test** — random neighbours must *actively hurt* performance
  relative to no retrieval at all. Target margin >= 0.041 nats. If random
  retrieval is neutral, the model is reading from its own weights rather
  than the bank, and the "Reader" claim has failed.

* **Semantic Gap** — real retrieval must lower validation loss by at least
  0.167 nats vs no retrieval. This is the headline number from the paper.

A third, project-level gate is the absolute floor:

* **Loss ceiling** — ``val_with_real`` must come in under 3.78 nats.

These functions take the per-mode mean losses (already produced by
``src/retro/eval_retro_heldout.py``) and stamp pass/fail. They are pure;
all I/O lives in ``scripts/run_acceptance_tests.py``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Mapping


# Roadmap thresholds. Kept here so a single edit retunes the gates.
IGNORANCE_HURT_MIN_NATS = 0.041
SEMANTIC_GAP_MIN_NATS = 0.167
VAL_WITH_LOSS_CEILING_NATS = 3.78

REQUIRED_MODES = ("none", "random", "real")


@dataclass(frozen=True)
class AcceptanceResult:
    """Outcome of one named gate."""

    name: str
    measured_nats: float
    threshold_nats: float
    direction: str  # "ge" (measured must be >= threshold) or "le"
    passed: bool
    detail: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AcceptanceReport:
    """Composite report covering all three reading-engine gates."""

    losses: Dict[str, float]
    gates: Dict[str, AcceptanceResult]
    all_passed: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "losses": dict(self.losses),
            "gates": {k: v.to_dict() for k, v in self.gates.items()},
            "all_passed": self.all_passed,
        }


def _require_modes(losses: Mapping[str, float]) -> None:
    missing = [m for m in REQUIRED_MODES if m not in losses]
    if missing:
        raise ValueError(
            "losses missing required modes "
            + ", ".join(missing)
            + f"; got {sorted(losses)}"
        )
    for mode in REQUIRED_MODES:
        v = losses[mode]
        if v != v or v <= 0:  # NaN or non-positive nats
            raise ValueError(f"loss[{mode}] is not a positive finite nats value: {v!r}")


def evaluate_ignorance_test(
    losses: Mapping[str, float],
    *,
    threshold_nats: float = IGNORANCE_HURT_MIN_NATS,
) -> AcceptanceResult:
    """Random neighbours should hurt vs no-retrieval by >= threshold nats.

    Sign convention: positive ``hurt_nats`` means ``loss(random) > loss(without)``.
    A near-zero (or negative) value means random retrieval is effectively
    neutral / helpful, which means the model is *not* truly reading from the
    bank for the random condition.

    Numeric trace (from the user-memory diagnostic rule):
        loss(without) = 4.000, loss(random) = 4.041
        hurt_nats = 4.041 - 4.000 = 0.041
        0.041 >= 0.041  ->  passes by the smallest legal margin
    """
    _require_modes(losses)
    hurt_nats = float(losses["random"] - losses["none"])
    passed = hurt_nats >= threshold_nats
    detail = (
        f"loss(random)={losses['random']:.4f} - loss(without)={losses['none']:.4f}"
        f" = {hurt_nats:+.4f} nats"
    )
    return AcceptanceResult(
        name="ignorance_test",
        measured_nats=hurt_nats,
        threshold_nats=threshold_nats,
        direction="ge",
        passed=passed,
        detail=detail,
    )


def evaluate_semantic_gap(
    losses: Mapping[str, float],
    *,
    threshold_nats: float = SEMANTIC_GAP_MIN_NATS,
) -> AcceptanceResult:
    """Real retrieval should lower loss vs no retrieval by >= threshold nats.

    Sign convention: positive ``gap_nats`` means ``loss(without) > loss(real)``.

    Numeric trace:
        loss(without) = 3.95, loss(real) = 3.78
        gap_nats = 3.95 - 3.78 = 0.17
        0.17 >= 0.167  ->  passes
    """
    _require_modes(losses)
    gap_nats = float(losses["none"] - losses["real"])
    passed = gap_nats >= threshold_nats
    detail = (
        f"loss(without)={losses['none']:.4f} - loss(real)={losses['real']:.4f}"
        f" = {gap_nats:+.4f} nats"
    )
    return AcceptanceResult(
        name="semantic_gap",
        measured_nats=gap_nats,
        threshold_nats=threshold_nats,
        direction="ge",
        passed=passed,
        detail=detail,
    )


def evaluate_loss_ceiling(
    losses: Mapping[str, float],
    *,
    ceiling_nats: float = VAL_WITH_LOSS_CEILING_NATS,
) -> AcceptanceResult:
    """val_with(real) must be at or below ceiling.

    Numeric trace:
        loss(real) = 3.78, ceiling = 3.78
        3.78 <= 3.78  ->  passes at the limit (any drift to 3.79 fails)
    """
    _require_modes(losses)
    real = float(losses["real"])
    passed = real <= ceiling_nats
    detail = f"loss(real)={real:.4f} <= ceiling={ceiling_nats:.4f}"
    return AcceptanceResult(
        name="loss_ceiling",
        measured_nats=real,
        threshold_nats=ceiling_nats,
        direction="le",
        passed=passed,
        detail=detail,
    )


def evaluate_all(losses: Mapping[str, float]) -> AcceptanceReport:
    """Run every gate and bundle the result. Pure; safe to call in tests."""
    gates = {
        "ignorance_test": evaluate_ignorance_test(losses),
        "semantic_gap": evaluate_semantic_gap(losses),
        "loss_ceiling": evaluate_loss_ceiling(losses),
    }
    return AcceptanceReport(
        losses=dict(losses),
        gates=gates,
        all_passed=all(g.passed for g in gates.values()),
    )
