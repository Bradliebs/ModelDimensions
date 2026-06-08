"""Per-layer attribution of the retrieval benefit.

The roadmap states that retrieval is concentrated in the middle of the
network, with one layer (layer 5 in the paper's setup) doing roughly 63%
of the work. We need a way to *measure* that, not assume it.

Mechanic
========

Run the held-out 3-condition eval one extra time per CCA layer with that
layer's cross-attention contribution zeroed out. The increase in
``val_with_real`` loss when layer ``L`` is suppressed is layer ``L``'s
*marginal contribution* to the retrieval benefit. Normalise by the total
benefit ``loss(without) - loss(real)`` and you get a per-layer share.

This module only does the math: it consumes a dict of per-layer
suppressed-real losses (which the CLI obtains by running the model in
ablated mode) and produces a normalised attribution table with explicit
pass/fail against a configurable concentration target.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Mapping


# Roadmap target: one layer in the CCA stack carries >= 0.50 of the share
# (the paper measures ~0.63). The wider target catches drift without
# triggering on small reshuffles between adjacent layers.
DOMINANT_LAYER_MIN_SHARE = 0.50


@dataclass(frozen=True)
class LayerContribution:
    layer_index: int
    suppressed_loss_real: float
    delta_vs_full_real: float  # nats added when this layer is suppressed
    share_of_total_benefit: float  # delta / total_benefit


@dataclass(frozen=True)
class LayerAttributionReport:
    losses: Dict[str, float]
    full_benefit_nats: float
    contributions: List[LayerContribution]
    dominant_layer_index: int
    dominant_layer_share: float
    dominant_share_target: float
    dominant_layer_passes: bool
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "losses": dict(self.losses),
            "full_benefit_nats": self.full_benefit_nats,
            "contributions": [asdict(c) for c in self.contributions],
            "dominant_layer_index": self.dominant_layer_index,
            "dominant_layer_share": self.dominant_layer_share,
            "dominant_share_target": self.dominant_share_target,
            "dominant_layer_passes": self.dominant_layer_passes,
            "notes": list(self.notes),
        }


def compute_layer_attribution(
    *,
    loss_without: float,
    loss_real_full: float,
    loss_real_with_layer_suppressed: Mapping[int, float],
    dominant_share_target: float = DOMINANT_LAYER_MIN_SHARE,
) -> LayerAttributionReport:
    """Turn per-layer suppression losses into a normalised attribution table.

    Sign convention:
      benefit  = loss_without - loss_real_full       (>= 0 expected)
      delta_L  = loss_with_L_suppressed - loss_real_full   (>= 0 expected)
      share_L  = delta_L / benefit

    If the full benefit is non-positive, attribution is meaningless — we
    surface this in ``notes`` and refuse to compute a dominant share.

    Numeric trace:
      loss_without=4.000, loss_real_full=3.800  ->  benefit = 0.200
      L5 suppressed -> 3.926, delta = 0.126, share = 0.63
      0.63 >= 0.50  ->  layer 5 dominant, passes
    """
    if not loss_real_with_layer_suppressed:
        raise ValueError("loss_real_with_layer_suppressed cannot be empty")
    for label, v in [("loss_without", loss_without), ("loss_real_full", loss_real_full)]:
        if v != v or v <= 0:
            raise ValueError(f"{label} must be positive finite nats; got {v!r}")
    for layer, v in loss_real_with_layer_suppressed.items():
        if not isinstance(layer, int):
            raise TypeError(
                f"layer indices must be int; got {type(layer).__name__} for {layer!r}"
            )
        if v != v or v <= 0:
            raise ValueError(
                f"loss_real_with_layer_suppressed[{layer}] must be positive finite; got {v!r}"
            )

    notes: List[str] = []
    benefit = loss_without - loss_real_full
    if benefit <= 0:
        notes.append(
            f"full retrieval benefit is non-positive ({benefit:+.4f} nats); attribution "
            "shares are meaningless and the dominant-layer gate cannot be evaluated."
        )

    contributions: List[LayerContribution] = []
    for layer in sorted(loss_real_with_layer_suppressed):
        suppressed = float(loss_real_with_layer_suppressed[layer])
        delta = suppressed - loss_real_full
        share = (delta / benefit) if benefit > 0 else 0.0
        if delta < 0:
            notes.append(
                f"layer {layer}: suppressing it *lowered* loss by "
                f"{-delta:+.4f} nats; this layer is hurting retrieval, not helping."
            )
        contributions.append(
            LayerContribution(
                layer_index=layer,
                suppressed_loss_real=suppressed,
                delta_vs_full_real=delta,
                share_of_total_benefit=share,
            )
        )

    if benefit > 0:
        dominant = max(contributions, key=lambda c: c.share_of_total_benefit)
        dominant_idx = dominant.layer_index
        dominant_share = dominant.share_of_total_benefit
        passes = dominant_share >= dominant_share_target
    else:
        dominant_idx = -1
        dominant_share = 0.0
        passes = False

    return LayerAttributionReport(
        losses={"without": loss_without, "real_full": loss_real_full},
        full_benefit_nats=benefit,
        contributions=contributions,
        dominant_layer_index=dominant_idx,
        dominant_layer_share=dominant_share,
        dominant_share_target=dominant_share_target,
        dominant_layer_passes=passes,
        notes=notes,
    )


def render_attribution_markdown(report: LayerAttributionReport) -> str:
    lines: List[str] = []
    lines.append("# Layer attribution of retrieval benefit")
    lines.append("")
    lines.append(
        f"- loss(without) = **{report.losses['without']:.4f}** nats"
    )
    lines.append(
        f"- loss(real, all CCA layers active) = **{report.losses['real_full']:.4f}** nats"
    )
    lines.append(
        f"- full benefit = **{report.full_benefit_nats:+.4f}** nats"
    )
    lines.append("")
    lines.append("| Layer | Loss when suppressed | Δ vs full real | Share of benefit |")
    lines.append("| ---: | ---: | ---: | ---: |")
    for c in report.contributions:
        lines.append(
            f"| {c.layer_index} | {c.suppressed_loss_real:.4f} | "
            f"{c.delta_vs_full_real:+.4f} | {c.share_of_total_benefit:+.3f} |"
        )
    lines.append("")
    lines.append(
        f"Dominant layer: **{report.dominant_layer_index}** "
        f"(share {report.dominant_layer_share:.3f}; target "
        f"{report.dominant_share_target:.2f})"
    )
    lines.append(
        f"Dominant-layer gate: **{'PASS' if report.dominant_layer_passes else 'FAIL'}**"
    )
    if report.notes:
        lines.append("")
        lines.append("## Notes")
        for n in report.notes:
            lines.append(f"- {n}")
    return "\n".join(lines) + "\n"
