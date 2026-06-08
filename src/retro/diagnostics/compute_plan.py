"""Compute and data-scaling plan for the Knowledge-Free Reader.

The roadmap asks for the first command:

    "Identify a high-quality, 8-billion-token dataset that can be split to
    ensure zero overlap with our existing 1.8M-cell Wikipedia bank, and
    outline the compute requirements for 400,000 iterations using 8-bit
    Adam."

This module answers the second half (compute + iteration math) as a pure
function. The first half (candidate datasets) is data, not arithmetic, and
lives in ``CHINCHILLA_CANDIDATES`` below. The disjointness *verification*
of any chosen shard is in ``disjointness.py``.

Numbers are deliberately conservative. Real throughput will swing 30-50%
depending on attention kernel, dataloader, and IO. The estimator returns
ranges, not point values, so the operator can't mis-quote them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence


# Chinchilla optimal ratio (Hoffmann et al., 2022): ~20 tokens per parameter.
CHINCHILLA_TOKENS_PER_PARAM = 20.0

# Throughput estimates for a single device, 404M dense + RETRO cross-attention,
# seq=256 / chunk=64 / n_neighbors=2 / neighbor_len=64. Tokens per second,
# total (not per-sample). Conservative lower / nominal upper bands.
# Sources: rough scaling from nanoGPT 124M ~80k tok/s on A100 dense baseline,
# discounted ~3x for the 404M depth and CCA overhead.
DEVICE_THROUGHPUT_TPS: Dict[str, Dict[str, float]] = {
    "rtx_3070_8gb_bf16": {"low": 1_500, "high": 3_500},
    "a100_40gb_bf16": {"low": 18_000, "high": 35_000},
    "a100_80gb_bf16": {"low": 22_000, "high": 45_000},
    "h100_80gb_bf16": {"low": 50_000, "high": 95_000},
}


@dataclass(frozen=True)
class DatasetCandidate:
    """A candidate corpus shard for the disjoint training mix."""

    name: str
    estimated_tokens: int
    license: str
    notes: str
    disjointness_risk: str  # "low" / "medium" / "high" vs Wikipedia bank


# Candidates for the 8B-token disjoint corpus. Selection rationale:
# - Avoid Wikipedia entirely (bank source) unless an article-level disjoint
#   slice is enforced via disjointness.py.
# - Prefer permissive licences so training artifacts can be released.
# - Mix conversation + code + clean web to give the model varied "reading"
#   surface without leaking the encyclopedic facts that live in the bank.
CHINCHILLA_CANDIDATES: List[DatasetCandidate] = [
    DatasetCandidate(
        name="OpenAssistant Conversations v2 (en)",
        estimated_tokens=80_000_000,
        license="Apache-2.0",
        notes=(
            "High-quality multi-turn instruction/conversation data. Too small "
            "alone (~80M tok) but anchors the 'reading skill' surface."
        ),
        disjointness_risk="low",
    ),
    DatasetCandidate(
        name="The Stack v2 dedup (permissive subset, 1% sample)",
        estimated_tokens=3_000_000_000,
        license="permissive (per-repo)",
        notes=(
            "Permissively licensed code, sampled at 1% from the dedup pool. "
            "Useful as 'symbolic reading' material; cannot leak Wikipedia "
            "factual content."
        ),
        disjointness_risk="low",
    ),
    DatasetCandidate(
        name="RedPajama v2 (sample, English, head bucket)",
        estimated_tokens=4_500_000_000,
        license="ODC-By 1.0",
        notes=(
            "Web text, deduplicated. MAY contain Wikipedia copies in the "
            "wild; require disjointness verification at paragraph SHA level "
            "before use. Filter URLs containing wikipedia.org."
        ),
        disjointness_risk="medium",
    ),
    DatasetCandidate(
        name="C4 cleaned-en (slice)",
        estimated_tokens=500_000_000,
        license="ODC-By 1.0",
        notes=(
            "Cleaned Common Crawl. Same Wikipedia-mirror caveat as RedPajama. "
            "Useful as filler if the others come up short."
        ),
        disjointness_risk="medium",
    ),
]


@dataclass(frozen=True)
class ComputeEstimate:
    """Output of ``estimate_compute_for_training``."""

    device: str
    throughput_low_tps: float
    throughput_high_tps: float
    hours_low: float
    hours_high: float
    dollars_low: float | None
    dollars_high: float | None


@dataclass(frozen=True)
class TrainingPlan:
    """Full plan: tokens, iterations, candidate mix, compute envelope."""

    n_params: int
    chinchilla_tokens: int
    unique_tokens_target: int
    iters: int
    effective_batch_size: int
    block_size: int
    presented_tokens: int
    presentations_per_unique_token: float
    candidates: List[DatasetCandidate]
    candidates_total_tokens: int
    coverage_ratio_unique_to_candidates: float
    coverage_ratio_presented_to_chinchilla: float
    compute_estimates: List[ComputeEstimate]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "n_params": self.n_params,
            "chinchilla_tokens": self.chinchilla_tokens,
            "unique_tokens_target": self.unique_tokens_target,
            "iters": self.iters,
            "effective_batch_size": self.effective_batch_size,
            "block_size": self.block_size,
            "presented_tokens": self.presented_tokens,
            "presentations_per_unique_token": self.presentations_per_unique_token,
            "candidates": [c.__dict__ for c in self.candidates],
            "candidates_total_tokens": self.candidates_total_tokens,
            "coverage_ratio_unique_to_candidates": self.coverage_ratio_unique_to_candidates,
            "coverage_ratio_presented_to_chinchilla": self.coverage_ratio_presented_to_chinchilla,
            "compute_estimates": [c.__dict__ for c in self.compute_estimates],
            "notes": list(self.notes),
        }


def chinchilla_token_target(n_params: int, *, ratio: float = CHINCHILLA_TOKENS_PER_PARAM) -> int:
    """Tokens per Chinchilla scaling law. 404M params * 20 = ~8.08B tokens."""
    if n_params <= 0:
        raise ValueError(f"n_params must be positive; got {n_params}")
    return int(round(n_params * ratio))


def presented_tokens(*, iters: int, effective_batch: int, block_size: int) -> int:
    """How many token-positions the optimiser actually sees across training."""
    for label, v in [("iters", iters), ("effective_batch", effective_batch),
                     ("block_size", block_size)]:
        if v <= 0:
            raise ValueError(f"{label} must be positive; got {v}")
    return int(iters) * int(effective_batch) * int(block_size)


def estimate_compute_for_training(
    *,
    presented_tokens_total: int,
    device: str,
    hourly_cost_usd: float | None = None,
) -> ComputeEstimate:
    """Wall-clock range and (optional) dollar range for one full training run."""
    if device not in DEVICE_THROUGHPUT_TPS:
        raise ValueError(
            f"unknown device {device!r}; expected one of {sorted(DEVICE_THROUGHPUT_TPS)}"
        )
    band = DEVICE_THROUGHPUT_TPS[device]
    low, high = band["low"], band["high"]
    hours_high = presented_tokens_total / low / 3600.0
    hours_low = presented_tokens_total / high / 3600.0
    dollars_low: float | None = None
    dollars_high: float | None = None
    if hourly_cost_usd is not None:
        if hourly_cost_usd < 0:
            raise ValueError(f"hourly_cost_usd must be non-negative; got {hourly_cost_usd}")
        dollars_low = hours_low * hourly_cost_usd
        dollars_high = hours_high * hourly_cost_usd
    return ComputeEstimate(
        device=device,
        throughput_low_tps=low,
        throughput_high_tps=high,
        hours_low=hours_low,
        hours_high=hours_high,
        dollars_low=dollars_low,
        dollars_high=dollars_high,
    )


def build_training_plan(
    *,
    n_params: int = 404_000_000,
    iters: int = 400_000,
    effective_batch: int = 32,
    block_size: int = 256,
    unique_tokens_target: int = 8_000_000_000,
    candidates: Sequence[DatasetCandidate] = tuple(CHINCHILLA_CANDIDATES),
    devices: Sequence[str] = ("a100_80gb_bf16", "h100_80gb_bf16"),
    hourly_costs_usd: Mapping[str, float] | None = None,
) -> TrainingPlan:
    """Build the full plan structure asked for by the roadmap's first command."""
    if not candidates:
        raise ValueError("candidates must be a non-empty sequence")
    presented = presented_tokens(
        iters=iters, effective_batch=effective_batch, block_size=block_size
    )
    candidates_total = sum(c.estimated_tokens for c in candidates)
    if candidates_total <= 0:
        raise ValueError("candidates produce zero total tokens")

    compute = []
    for device in devices:
        hc = None if hourly_costs_usd is None else hourly_costs_usd.get(device)
        compute.append(
            estimate_compute_for_training(
                presented_tokens_total=presented,
                device=device,
                hourly_cost_usd=hc,
            )
        )

    notes: List[str] = []
    chinchilla = chinchilla_token_target(n_params)
    if unique_tokens_target < chinchilla:
        notes.append(
            f"unique_tokens_target {unique_tokens_target:,} is below the Chinchilla "
            f"optimum of {chinchilla:,} tokens for {n_params:,} params; expect a small "
            f"loss penalty."
        )
    if presented < unique_tokens_target:
        notes.append(
            f"presented_tokens {presented:,} is below the unique-token target "
            f"{unique_tokens_target:,}; the model will see less than one epoch. "
            f"Increase iters or effective_batch to cover the corpus."
        )
    if candidates_total < unique_tokens_target:
        notes.append(
            f"candidates_total {candidates_total:,} is below unique_tokens_target "
            f"{unique_tokens_target:,}; the disjoint corpus is too small."
        )
    if presented > 3 * unique_tokens_target:
        notes.append(
            f"presented_tokens {presented:,} is more than 3x unique_tokens_target "
            f"{unique_tokens_target:,}; multi-epoch repetition is likely. The "
            f"'Reader' hypothesis assumes mostly-fresh tokens; revisit."
        )

    return TrainingPlan(
        n_params=n_params,
        chinchilla_tokens=chinchilla,
        unique_tokens_target=unique_tokens_target,
        iters=iters,
        effective_batch_size=effective_batch,
        block_size=block_size,
        presented_tokens=presented,
        presentations_per_unique_token=presented / unique_tokens_target,
        candidates=list(candidates),
        candidates_total_tokens=candidates_total,
        coverage_ratio_unique_to_candidates=unique_tokens_target / candidates_total,
        coverage_ratio_presented_to_chinchilla=presented / chinchilla,
        compute_estimates=compute,
        notes=notes,
    )


def render_plan_markdown(plan: TrainingPlan) -> str:
    """Deterministic markdown summary; no I/O."""
    lines: List[str] = []
    lines.append("# Chinchilla scaling plan — Knowledge-Free Reader")
    lines.append("")
    lines.append("## Parameter and token targets")
    lines.append("")
    lines.append(f"- Model parameters: **{plan.n_params:,}**")
    lines.append(
        f"- Chinchilla-optimal tokens (20x params): **{plan.chinchilla_tokens:,}**"
    )
    lines.append(f"- Unique-token target for this plan: **{plan.unique_tokens_target:,}**")
    lines.append(
        f"- Iterations: **{plan.iters:,}** at effective batch "
        f"**{plan.effective_batch_size}**, block size **{plan.block_size}**"
    )
    lines.append(f"- Token-presentations across the run: **{plan.presented_tokens:,}**")
    lines.append(
        f"- Presentations per unique token: "
        f"**{plan.presentations_per_unique_token:.2f}**"
    )
    lines.append(
        f"- Coverage ratio (presented / Chinchilla): "
        f"**{plan.coverage_ratio_presented_to_chinchilla:.2f}**"
    )
    lines.append("")
    lines.append("## Candidate disjoint corpora")
    lines.append("")
    lines.append("| Source | Est. tokens | Licence | Disjointness risk |")
    lines.append("| --- | ---: | --- | --- |")
    for c in plan.candidates:
        lines.append(
            f"| {c.name} | {c.estimated_tokens:,} | {c.license} | {c.disjointness_risk} |"
        )
    lines.append(
        f"| **Total** | **{plan.candidates_total_tokens:,}** | | |"
    )
    lines.append("")
    lines.append("## Compute envelope")
    lines.append("")
    lines.append("| Device | Throughput tok/s (low–high) | Hours (low–high) | USD (low–high) |")
    lines.append("| --- | --- | --- | --- |")
    for ce in plan.compute_estimates:
        cost = (
            f"${ce.dollars_low:,.0f} – ${ce.dollars_high:,.0f}"
            if ce.dollars_low is not None
            else "(no rate supplied)"
        )
        lines.append(
            f"| {ce.device} | {ce.throughput_low_tps:,.0f} – "
            f"{ce.throughput_high_tps:,.0f} | "
            f"{ce.hours_low:.1f} – {ce.hours_high:.1f} | {cost} |"
        )
    if plan.notes:
        lines.append("")
        lines.append("## Notes and warnings")
        lines.append("")
        for n in plan.notes:
            lines.append(f"- {n}")
    return "\n".join(lines) + "\n"
