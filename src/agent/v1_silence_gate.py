"""V1 silence gate: separates real hits from noise on the production bank.

Chosen by `experiments/exp16_gate_signals.py` after a head-to-head sweep
over candidate signals on 50 probe queries (20 known, 20 unknown, 10
gibberish-noise) against the 5.7M-cell bank. The headline finding was that
the existing `rerank_min_score = 0.10` threshold cleared 70% of noise
queries — almost no discrimination at the boundary. The chosen gate is
the per-query concentration of the top-1 hit relative to the second hit:

    fire IF (top-1 activation - top-2 activation) >= 0.05

On the same 50 queries:
    known   85%   (vs 85% with the old gate)
    unknown  0%   (vs 25% with the old gate)
    noise   10%   (vs 70% with the old gate)

self-reference cosine (encode top-1's text, compare to query) looked even
stronger but was inflated by the known-query test design (queries were
literal first-sentences from `source_texts`). It is intentionally NOT
used here; once realistic-question validation exists it can be added as a
second-stage gate.

The gate is a single value with a single threshold. No tunable weights,
no hidden state. If you need to override the threshold (e.g. for an
expert library where false-positives are worth tolerating), pass
``margin_threshold`` explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

# Margin threshold (top-1 - top-2 activation) chosen from exp16 sweep.
# Lower values let noise through; higher values silence real hits.
DEFAULT_MARGIN_THRESHOLD: float = 0.05


@dataclass
class SilenceDecision:
    """Verdict from the silence gate."""

    fire: bool
    top1_activation: float
    top2_activation: float
    margin: float
    threshold: float
    reason: str

    def as_dict(self) -> dict:
        return {
            "fire": self.fire,
            "top1_activation": self.top1_activation,
            "top2_activation": self.top2_activation,
            "margin": self.margin,
            "threshold": self.threshold,
            "reason": self.reason,
        }


def decide(activations: Sequence[float],
           margin_threshold: float = DEFAULT_MARGIN_THRESHOLD
           ) -> SilenceDecision:
    """Decide whether retrieval is concentrated enough to ground an answer.

    ``activations`` is the descending-sorted top-k activation list from
    :meth:`StreamingBank.topk`. We need at least the top two values; if
    fewer cells are available (e.g. tiny test bank), the gate fires only
    when the single available activation is itself well above zero.
    """
    if len(activations) == 0:
        return SilenceDecision(
            fire=False,
            top1_activation=float("nan"),
            top2_activation=float("nan"),
            margin=float("nan"),
            threshold=margin_threshold,
            reason="no retrieval results",
        )

    top1 = float(activations[0])

    if len(activations) == 1:
        # Tiny bank fallback: insist top-1 alone is meaningfully positive.
        if top1 >= margin_threshold:
            return SilenceDecision(
                fire=True,
                top1_activation=top1,
                top2_activation=float("nan"),
                margin=top1,
                threshold=margin_threshold,
                reason="single-cell bank, top1 above threshold",
            )
        return SilenceDecision(
            fire=False,
            top1_activation=top1,
            top2_activation=float("nan"),
            margin=top1,
            threshold=margin_threshold,
            reason="single-cell bank, top1 below threshold",
        )

    top2 = float(activations[1])
    margin = top1 - top2
    fire = bool(margin >= margin_threshold)
    return SilenceDecision(
        fire=fire,
        top1_activation=top1,
        top2_activation=top2,
        margin=margin,
        threshold=margin_threshold,
        reason=("margin above threshold"
                if fire
                else "margin below threshold (retrieval diffuse)"),
    )


__all__ = ["SilenceDecision", "decide", "DEFAULT_MARGIN_THRESHOLD"]
