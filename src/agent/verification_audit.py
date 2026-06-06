"""V1.1 verification-audit utilities.

This module is deliberately evaluation-only. It does not change retrieval,
generation, verifier thresholds, or answer policy. It gives V1 a more honest
way to report safety claims: counts, Wilson intervals, zero-event upper bounds,
risk/coverage curves, detector ablations, and differential comparisons against
an external NLI-style label.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from slm.schemas import VerificationVerdict

from src.agent import v1_answer_verifier, verifier


EXPECTED_ACCEPT = "accept"
EXPECTED_REJECT = "reject"
NLI_ENTAILMENT = "entailment"
NLI_CONTRADICTION = "contradiction"
NLI_NEUTRAL = "neutral"


@dataclass(frozen=True)
class WilsonInterval:
    lower: float
    upper: float

    def as_dict(self) -> dict:
        return {"lower": self.lower, "upper": self.upper}


@dataclass(frozen=True)
class BinaryMetric:
    name: str
    positive: int
    total: int
    positive_label: str
    negative_label: str
    wilson_95: WilsonInterval
    rule_of_three_upper_bound: float | None = None

    @property
    def negative(self) -> int:
        return self.total - self.positive

    @property
    def observed_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.positive / self.total

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "positive": self.positive,
            "negative": self.negative,
            "total": self.total,
            "positive_label": self.positive_label,
            "negative_label": self.negative_label,
            "observed_rate": self.observed_rate,
            "wilson_95": self.wilson_95.as_dict(),
            "rule_of_three_upper_bound": self.rule_of_three_upper_bound,
        }


@dataclass(frozen=True)
class CandidateAuditCase:
    case_id: str
    evidence: str
    claim: str
    expected_label: str
    category: str
    nli_label: str = ""
    relation_id: str = ""
    relation_role: str = ""
    notes: str = ""

    @classmethod
    def from_dict(cls, raw: dict) -> "CandidateAuditCase":
        expected = str(raw.get("expected_label", "")).strip().lower()
        if expected not in {EXPECTED_ACCEPT, EXPECTED_REJECT}:
            raise ValueError(
                f"case {raw.get('case_id', '<missing>')!r} has invalid "
                f"expected_label {expected!r}"
            )
        case_id = str(raw.get("case_id", "")).strip()
        if not case_id:
            raise ValueError("audit case is missing case_id")
        return cls(
            case_id=case_id,
            evidence=str(raw.get("evidence", "")),
            claim=str(raw.get("claim", "")),
            expected_label=expected,
            category=str(raw.get("category", "uncategorized")),
            nli_label=str(raw.get("nli_label", "")).strip().lower(),
            relation_id=str(raw.get("relation_id", "")),
            relation_role=str(raw.get("relation_role", "")),
            notes=str(raw.get("notes", "")),
        )

    def as_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "evidence": self.evidence,
            "claim": self.claim,
            "expected_label": self.expected_label,
            "category": self.category,
            "nli_label": self.nli_label,
            "relation_id": self.relation_id,
            "relation_role": self.relation_role,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class CandidateAuditObservation:
    case: CandidateAuditCase
    verdict: str
    answered: bool
    safe: bool
    false_accept: bool
    over_refusal: bool
    nli_agreement: str

    def as_dict(self) -> dict:
        return {
            "case_id": self.case.case_id,
            "category": self.case.category,
            "expected_label": self.case.expected_label,
            "verdict": self.verdict,
            "answered": self.answered,
            "safe": self.safe,
            "false_accept": self.false_accept,
            "over_refusal": self.over_refusal,
            "nli_label": self.case.nli_label,
            "nli_agreement": self.nli_agreement,
        }


@dataclass(frozen=True)
class RiskCoveragePoint:
    threshold: float
    answered: int
    total: int
    false_accepts: int
    correct_accepts: int

    @property
    def coverage_rate(self) -> float:
        return self.answered / self.total if self.total else 0.0

    @property
    def error_rate_among_answered(self) -> float:
        return self.false_accepts / self.answered if self.answered else 0.0

    def as_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "answered": self.answered,
            "total": self.total,
            "coverage_rate": self.coverage_rate,
            "false_accepts": self.false_accepts,
            "correct_accepts": self.correct_accepts,
            "error_rate_among_answered": self.error_rate_among_answered,
        }


@dataclass(frozen=True)
class DetectorAblationResult:
    detector_name: str
    baseline_false_accepts: int
    ablated_false_accepts: int
    newly_accepted_case_ids: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "detector_name": self.detector_name,
            "baseline_false_accepts": self.baseline_false_accepts,
            "ablated_false_accepts": self.ablated_false_accepts,
            "newly_accepted_case_ids": list(self.newly_accepted_case_ids),
        }


@dataclass(frozen=True)
class VerificationAuditReport:
    cases: tuple[CandidateAuditCase, ...]
    observations: tuple[CandidateAuditObservation, ...]
    metrics: tuple[BinaryMetric, ...]
    risk_coverage_curve: tuple[RiskCoveragePoint, ...]
    detector_ablations: tuple[DetectorAblationResult, ...]

    def as_dict(self) -> dict:
        return {
            "case_count": len(self.cases),
            "cases": [case.as_dict() for case in self.cases],
            "observations": [obs.as_dict() for obs in self.observations],
            "metrics": [metric.as_dict() for metric in self.metrics],
            "risk_coverage_curve": [p.as_dict() for p in self.risk_coverage_curve],
            "detector_ablations": [a.as_dict() for a in self.detector_ablations],
        }


def wilson_interval(positive: int, total: int, z: float = 1.959963984540054) -> WilsonInterval:
    """Return a two-sided Wilson interval for a binomial proportion."""
    if total < 0 or positive < 0 or positive > total:
        raise ValueError("positive must be between 0 and total")
    if total == 0:
        return WilsonInterval(0.0, 0.0)
    phat = positive / total
    denom = 1 + (z * z / total)
    centre = phat + (z * z / (2 * total))
    spread = z * math.sqrt((phat * (1 - phat) + (z * z / (4 * total))) / total)
    return WilsonInterval(
        lower=max(0.0, (centre - spread) / denom),
        upper=min(1.0, (centre + spread) / denom),
    )


def rule_of_three_upper_bound(events: int, total: int) -> float | None:
    """Return the 95% zero-event upper bound, or None when events > 0."""
    if total <= 0 or events != 0:
        return None
    return min(1.0, 3.0 / total)


def metric(name: str, positive: int, total: int, positive_label: str, negative_label: str) -> BinaryMetric:
    return BinaryMetric(
        name=name,
        positive=positive,
        total=total,
        positive_label=positive_label,
        negative_label=negative_label,
        wilson_95=wilson_interval(positive, total),
        rule_of_three_upper_bound=rule_of_three_upper_bound(positive, total),
    )


def load_candidate_audit_cases(path: Path) -> list[CandidateAuditCase]:
    """Load JSONL audit cases. Blank lines and # comments are ignored."""
    cases: list[CandidateAuditCase] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            cases.append(CandidateAuditCase.from_dict(json.loads(stripped)))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
    return cases


def _verdict_name(verdict: VerificationVerdict) -> str:
    return str(verdict.value if hasattr(verdict, "value") else verdict).lower()


def _nli_agreement(expected_verdict: str, nli_label: str) -> str:
    if not nli_label:
        return "not_provided"
    if expected_verdict == "accept" and nli_label == NLI_ENTAILMENT:
        return "agreement"
    if expected_verdict == "reject" and nli_label == NLI_CONTRADICTION:
        return "agreement"
    if expected_verdict == "ambiguous" and nli_label == NLI_NEUTRAL:
        return "agreement"
    if expected_verdict == "accept" and nli_label == NLI_CONTRADICTION:
        return "inspect_urgently"
    if expected_verdict == "reject" and nli_label == NLI_ENTAILMENT:
        return "possible_over_refusal"
    return "review"


def observe_candidate_case(case: CandidateAuditCase) -> CandidateAuditObservation:
    verdict = verifier.verify_candidate(case.claim, case.evidence)
    verdict_name = _verdict_name(verdict)
    answered = verdict is VerificationVerdict.ACCEPT
    false_accept = answered and case.expected_label == EXPECTED_REJECT
    over_refusal = (not answered) and case.expected_label == EXPECTED_ACCEPT
    safe = not false_accept
    return CandidateAuditObservation(
        case=case,
        verdict=verdict_name,
        answered=answered,
        safe=safe,
        false_accept=false_accept,
        over_refusal=over_refusal,
        nli_agreement=_nli_agreement(verdict_name, case.nli_label),
    )


def summarize_observations(observations: Sequence[CandidateAuditObservation]) -> tuple[BinaryMetric, ...]:
    total = len(observations)
    answered = sum(1 for obs in observations if obs.answered)
    false_accepts = sum(1 for obs in observations if obs.false_accept)
    reject_cases = sum(1 for obs in observations if obs.case.expected_label == EXPECTED_REJECT)
    accept_cases = sum(1 for obs in observations if obs.case.expected_label == EXPECTED_ACCEPT)
    over_refusals = sum(1 for obs in observations if obs.over_refusal)
    answered_correct = sum(
        1 for obs in observations
        if obs.answered and obs.case.expected_label == EXPECTED_ACCEPT
    )
    return (
        metric("coverage", answered, total, "answered", "not_answered"),
        metric("false_accept_rate", false_accepts, reject_cases, "false_accept", "safe_reject"),
        metric("over_refusal_rate", over_refusals, accept_cases, "over_refusal", "accepted"),
        metric("accuracy_among_answered", answered_correct, answered, "correct_answer", "wrong_answer"),
    )


def risk_coverage_curve_for_answer_verifier(
    cases: Sequence[CandidateAuditCase],
    thresholds: Iterable[float],
) -> tuple[RiskCoveragePoint, ...]:
    """Evaluate V1 answer verifier coverage/error tradeoff by threshold."""
    points: list[RiskCoveragePoint] = []
    for threshold in thresholds:
        answered = 0
        false_accepts = 0
        correct_accepts = 0
        for case in cases:
            decision = v1_answer_verifier.verify(
                case.claim,
                [case.evidence],
                question="",
                min_coverage=float(threshold),
                strict_nonsense=False,
            )
            if not decision.grounded:
                continue
            answered += 1
            if case.expected_label == EXPECTED_REJECT:
                false_accepts += 1
            else:
                correct_accepts += 1
        points.append(
            RiskCoveragePoint(
                threshold=float(threshold),
                answered=answered,
                total=len(cases),
                false_accepts=false_accepts,
                correct_accepts=correct_accepts,
            )
        )
    return tuple(points)


def _verify_candidate_with_disabled_detectors(
    claim: str,
    evidence: str,
    disabled_detector_names: set[str],
) -> VerificationVerdict:
    for detector in verifier._DETECTORS:  # noqa: SLF001 - internal audit of internal verifier.
        if detector.__name__ in disabled_detector_names:
            continue
        if detector(claim, evidence):
            return VerificationVerdict.REJECT

    if verifier._norm_key(claim) == verifier._norm_key(evidence):  # noqa: SLF001
        return VerificationVerdict.ACCEPT
    if verifier._strong_containment(claim, evidence):  # noqa: SLF001
        return VerificationVerdict.ACCEPT
    if verifier._is_fixture_paraphrase(claim, evidence):  # noqa: SLF001
        return VerificationVerdict.ACCEPT
    return VerificationVerdict.AMBIGUOUS


def detector_ablation_results(cases: Sequence[CandidateAuditCase]) -> tuple[DetectorAblationResult, ...]:
    baseline_false_accepts = sum(
        1 for case in cases
        if case.expected_label == EXPECTED_REJECT
        and verifier.verify_candidate(case.claim, case.evidence) is VerificationVerdict.ACCEPT
    )
    results: list[DetectorAblationResult] = []
    for detector in verifier._DETECTORS:  # noqa: SLF001 - detector inventory is the audit target.
        newly_accepted: list[str] = []
        ablated_false_accepts = 0
        for case in cases:
            if case.expected_label != EXPECTED_REJECT:
                continue
            baseline = verifier.verify_candidate(case.claim, case.evidence)
            ablated = _verify_candidate_with_disabled_detectors(
                case.claim,
                case.evidence,
                {detector.__name__},
            )
            if ablated is VerificationVerdict.ACCEPT:
                ablated_false_accepts += 1
            if baseline is not VerificationVerdict.ACCEPT and ablated is VerificationVerdict.ACCEPT:
                newly_accepted.append(case.case_id)
        results.append(
            DetectorAblationResult(
                detector_name=detector.__name__,
                baseline_false_accepts=baseline_false_accepts,
                ablated_false_accepts=ablated_false_accepts,
                newly_accepted_case_ids=tuple(newly_accepted),
            )
        )
    return tuple(results)


def run_verification_audit(
    cases: Sequence[CandidateAuditCase],
    thresholds: Sequence[float] = (0.35, 0.50, 0.65, 0.80),
) -> VerificationAuditReport:
    observations = tuple(observe_candidate_case(case) for case in cases)
    return VerificationAuditReport(
        cases=tuple(cases),
        observations=observations,
        metrics=summarize_observations(observations),
        risk_coverage_curve=risk_coverage_curve_for_answer_verifier(cases, thresholds),
        detector_ablations=detector_ablation_results(cases),
    )


def render_verification_audit_markdown(report: VerificationAuditReport) -> str:
    lines = ["# V1.1 verification audit", "", f"Cases: {len(report.cases)}", ""]
    lines.append("## Metrics")
    lines.append("| Metric | Events | Total | Observed | Wilson 95% | Rule-of-three upper |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for item in report.metrics:
        rot = item.rule_of_three_upper_bound
        rot_text = "" if rot is None else f"{rot:.3f}"
        lines.append(
            f"| {item.name} | {item.positive} | {item.total} | "
            f"{item.observed_rate:.3f} | "
            f"{item.wilson_95.lower:.3f}-{item.wilson_95.upper:.3f} | {rot_text} |"
        )
    lines.extend(["", "## Risk / Coverage Curve"])
    lines.append("| Coverage threshold | Answered | Coverage | False accepts | Error among answered |")
    lines.append("| ---: | ---: | ---: | ---: | ---: |")
    for point in report.risk_coverage_curve:
        lines.append(
            f"| {point.threshold:.2f} | {point.answered}/{point.total} | "
            f"{point.coverage_rate:.3f} | {point.false_accepts} | "
            f"{point.error_rate_among_answered:.3f} |"
        )
    lines.extend(["", "## Differential Review Candidates"])
    disagreements = [
        obs for obs in report.observations
        if obs.nli_agreement not in {"agreement", "not_provided"}
    ]
    if disagreements:
        lines.append("| Case | Verifier | NLI label | Meaning |")
        lines.append("| --- | --- | --- | --- |")
        for obs in disagreements:
            lines.append(
                f"| {obs.case.case_id} | {obs.verdict} | {obs.case.nli_label} | "
                f"{obs.nli_agreement} |"
            )
    else:
        lines.append("No verifier/NLI disagreements in the supplied labels.")
    lines.extend(["", "## Detector Ablations"])
    lines.append("| Disabled detector | Baseline false accepts | Ablated false accepts | Newly accepted rejects |")
    lines.append("| --- | ---: | ---: | --- |")
    for item in report.detector_ablations:
        newly = ", ".join(item.newly_accepted_case_ids)
        lines.append(
            f"| {item.detector_name} | {item.baseline_false_accepts} | "
            f"{item.ablated_false_accepts} | {newly} |"
        )
    return "\n".join(lines) + "\n"


def write_verification_audit(report: VerificationAuditReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True), encoding="utf-8")