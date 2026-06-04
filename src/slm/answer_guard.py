"""AnswerGuard: an independent, deterministic check over a composed answer.

This is the "immune system" of the assistant layer. Where the composer renders
a :class:`~slm.assistant_composer.GroundingPackage` into prose, the guard takes
the *finished* :class:`~slm.assistant_composer.ComposedAnswer` and verifies,
after the fact, that it did not drift from the package:

* no citation appears that is not in the package's allowed evidence;
* a non-grounded answer (refusal / conflict / model-prior) cites nothing;
* a refusal was not silently softened into an answer;
* a model-prior answer is labelled as such and never presents as grounded;
* the answer's mode matches the package's mode (no relabelling);
* a stale-source caution in the package is still surfaced in the prose.

Why this exists when ``LocalSLMComposer`` already validates its own output: the
SLM composer only guards *its own* path, as a silent fallback trigger. But
``WorkbenchService.answer_query`` / ``compose_answer`` accept an **arbitrary**
composer. A custom or future composer is not bound by that internal check. The
AnswerGuard is composer-agnostic and runs on every answer, producing an explicit
``ACCEPT`` / ``REJECT`` verdict that lands in the audit trail.

The guard is pure: :func:`check_answer` computes a verdict and never mutates
anything. :func:`enforce` is the safe wrapper — on ``REJECT`` it discards the
offending answer and recomposes deterministically from the same package with the
always-correct :class:`~slm.assistant_composer.TemplateComposer`. The built-in
template and SLM composers are correct by construction, so ``enforce`` only ever
changes output for a misbehaving third-party composer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List

from .assistant_composer import (
    _CITATION_RE,
    JUDGEMENT_LABEL,
    SECTION_FACTUAL,
    SECTION_JUDGEMENT,
    ComposedAnswer,
    ComposerMode,
    GroundingPackage,
    TemplateComposer,
)

# Modes for which citing a grounded source is legitimate.
_CITABLE_MODES = (ComposerMode.GROUNDED, ComposerMode.PACK_SUMMARY)

# Violation codes (stable strings so audits and tests can assert on them).
INVENTED_CITATION = "invented_citation"
CITATION_ON_NONGROUNDED = "citation_on_nongrounded"
SOFTENED_REFUSAL = "softened_refusal"
UNLABELLED_MODEL_PRIOR = "unlabelled_model_prior"
MODE_MISMATCH = "mode_mismatch"
DROPPED_STALE_CAUTION = "dropped_stale_caution"
UNSUPPORTED_SPAN = "unsupported_span"
# v2.7 consultant-report checks (no-op unless the answer carries a report).
UNLABELLED_JUDGEMENT = "unlabelled_judgement"
CITED_JUDGEMENT = "cited_judgement"
UNSUPPORTED_REPORT_CLAIM = "unsupported_report_claim"


@dataclass(frozen=True)
class GuardViolation:
    """One way the composed answer drifted from its grounding package."""

    code: str
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class GuardReport:
    """The verdict for one composed answer."""

    verdict: str  # "ACCEPT" | "REJECT"
    violations: List[GuardViolation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.verdict == "ACCEPT"

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "violations": [v.to_dict() for v in self.violations],
        }


def _cited_ids(answer: ComposedAnswer) -> set[str]:
    """All ids the answer cites: its structured list plus any text markers."""
    cited = {c.strip() for c in answer.citations if c.strip()}
    for kind, ident in _CITATION_RE.findall(answer.text or ""):
        cited.add(f"{kind}:{ident.strip()}")
    return cited


def _has_stale_caution(cautions: List[str]) -> bool:
    return any("stale" in c.lower() for c in cautions)


def check_answer(package: GroundingPackage,
                 answer: ComposedAnswer) -> GuardReport:
    """Verify ``answer`` against ``package``. Pure; never mutates either."""
    violations: List[GuardViolation] = []
    allowed = package.allowed_citation_ids
    cited = _cited_ids(answer)

    invented = cited - allowed
    if invented:
        violations.append(GuardViolation(
            INVENTED_CITATION,
            f"answer cites {sorted(invented)} not in allowed evidence "
            f"{sorted(allowed)}"))

    if cited and package.mode not in _CITABLE_MODES:
        violations.append(GuardViolation(
            CITATION_ON_NONGROUNDED,
            f"answer cites {sorted(cited)} on a non-grounded "
            f"{package.mode.value} answer"))

    if (package.refused or package.mode == ComposerMode.REFUSAL) \
            and not answer.refused:
        violations.append(GuardViolation(
            SOFTENED_REFUSAL,
            "package refused but the composed answer is not marked refused"))

    # A model-prior answer is identified by its mode. (``package.model_prior_used``
    # is a broader audit signal — it is also true on a plain refusal where
    # nothing grounded — so it must NOT be used here, or every refusal would be
    # flagged. Only mode MODEL_PRIOR_LABELLED is an actual model-prior answer.)
    if package.mode == ComposerMode.MODEL_PRIOR_LABELLED \
            and not answer.model_prior_labelled:
        violations.append(GuardViolation(
            UNLABELLED_MODEL_PRIOR,
            "model-prior answer is not labelled as a model prior"))

    if answer.mode != package.mode:
        violations.append(GuardViolation(
            MODE_MISMATCH,
            f"answer mode {answer.mode.value} != package mode "
            f"{package.mode.value}"))

    if _has_stale_caution(package.cautions) \
            and "stale" not in (answer.text or "").lower():
        violations.append(GuardViolation(
            DROPPED_STALE_CAUTION,
            "package carries a stale-source caution the answer does not "
            "surface"))

    # Span-level binding (v2.6): when a composer emits per-span citations, every
    # span must be a verbatim substring of the evidence item it cites. This
    # mechanically forbids unsupported bridging text. No-op for whole-chunk
    # composers, whose ``spans`` list is empty.
    if answer.spans:
        evidence_text = {e.citation_id: e.text for e in package.evidence}
        for span in answer.spans:
            source = evidence_text.get(span.citation_id)
            if source is None or span.text not in source:
                violations.append(GuardViolation(
                    UNSUPPORTED_SPAN,
                    f"span {span.text!r} cited to {span.citation_id} is not a "
                    "verbatim substring of that evidence item"))

    # Consultant-report partition (v2.7): the report carrier — not the prose — is
    # the source of truth for each section's kind. A *factual* section must carry
    # only citation-bound, verbatim spans; a *judgement* section must be labelled
    # and contain no citation marker. No-op when ``answer.report`` is None.
    report = getattr(answer, "report", None)
    if report is not None:
        evidence_text = {e.citation_id: e.text for e in package.evidence}
        for section in report.sections:
            if section.kind == SECTION_FACTUAL:
                for span in section.spans:
                    source = evidence_text.get(span.citation_id)
                    if source is None or span.text not in source:
                        violations.append(GuardViolation(
                            UNSUPPORTED_REPORT_CLAIM,
                            f"factual section {section.title!r} claim "
                            f"{span.text!r} cited to {span.citation_id} is not "
                            "a verbatim substring of that evidence item"))
            elif section.kind == SECTION_JUDGEMENT:
                judgement = section.judgement_text or ""
                if not judgement.startswith(JUDGEMENT_LABEL):
                    violations.append(GuardViolation(
                        UNLABELLED_JUDGEMENT,
                        f"judgement section {section.title!r} is not labelled "
                        f"with {JUDGEMENT_LABEL!r}"))
                if _CITATION_RE.search(judgement):
                    violations.append(GuardViolation(
                        CITED_JUDGEMENT,
                        f"judgement section {section.title!r} contains a "
                        "citation marker; judgement must be uncited"))

    verdict = "ACCEPT" if not violations else "REJECT"
    return GuardReport(verdict=verdict, violations=violations)


def enforce(package: GroundingPackage,
            answer: ComposedAnswer) -> tuple[ComposedAnswer, GuardReport]:
    """Return a safe answer plus its verdict.

    If ``answer`` passes, it is returned unchanged. If it fails, it is discarded
    and a deterministic template answer is composed from the same package — the
    template is correct by construction, so the result is always clean. The
    returned report is the verdict for the *original* answer, so the audit trail
    still records that a rejection happened.
    """
    report = check_answer(package, answer)
    if report.ok:
        return answer, report
    safe = TemplateComposer().compose(package)
    return safe, report
