"""Assistant composer: turn a grounded query result into a readable answer.

This is the optional "assistant layer" over the frozen v1.0 memory path. It
*composes prose*; it never decides what is true. Every load-bearing decision —
whether memory or knowledge was grounded, whether the query was refused, which
ids may be cited — is made deterministically upstream by
``WorkbenchService.build_grounding_package`` and frozen into a
:class:`GroundingPackage`. A composer only renders that package into words.

Two composers are provided:

* :class:`TemplateComposer` — the default. Deterministic, offline, no model.
* :class:`LocalSLMComposer` — opt-in. Wraps a local SLM backend to make the
  prose smoother, but is structurally prevented from changing any decision:

  - the answer ``mode``, the ``refused`` flag and the informational/model-prior
    flags are copied from the package, never read from the model;
  - the set of citable ids comes from the package; if the model emits a
    citation marker (``[mem:...]`` / ``[src:...]``) that is not in the allowed
    set, or cites anything at all on a refusal/conflict/model-prior answer, the
    output is rejected and the composer falls back to the template;
  - empty or malformed model output falls back to the template.

So the SLM can only ever make a *safe* answer read better. It can never invent
a citation, ground a refused query, or relabel an ungrounded answer.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from .local_slm_backend import LocalSLMBackend

# Citation markers the SLM is permitted to echo, e.g. ``[mem:m-3]`` or
# ``[src:chunk-7]``. Anything matching this shape that is not in the allowed
# set is treated as an invented citation.
_CITATION_RE = re.compile(r"\[(mem|src):([^\]]+)\]")


class ComposerMode(str, Enum):
    """The kind of answer being composed, decided upstream, not by a composer."""

    GROUNDED = "grounded"
    REFUSAL = "refusal"
    MODEL_PRIOR_LABELLED = "model_prior_labelled"
    CONFLICT_EXPLANATION = "conflict_explanation"
    PACK_SUMMARY = "pack_summary"


@dataclass(frozen=True)
class EvidenceItem:
    """One piece of allowed evidence with a stable citation id."""

    citation_id: str          # "mem:<memory_id>" or "src:<chunk_id>"
    kind: str                 # "memory" | "knowledge"
    text: str
    source_name: Optional[str] = None
    domain: Optional[str] = None
    authority: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "citation_id": self.citation_id,
            "kind": self.kind,
            "text": self.text,
            "source_name": self.source_name,
            "domain": self.domain,
            "authority": self.authority,
        }


@dataclass
class GroundingPackage:
    """The frozen, deterministic envelope a composer is allowed to render.

    ``evidence`` is the *only* citable content. ``conflict_context`` and
    ``historical_note`` are display-only: a near-miss that the verifier rejected,
    or a superseded memory, may be shown for transparency but can never be
    cited as a current answer.
    """

    query: str
    mode: ComposerMode
    memory_used: bool
    knowledge_used: bool
    model_prior_used: bool
    informational_only: bool
    refused: bool
    route: str
    evidence: List[EvidenceItem] = field(default_factory=list)
    conflict_context: List[EvidenceItem] = field(default_factory=list)
    cautions: List[str] = field(default_factory=list)
    historical_note: Optional[str] = None
    query_audit: Optional[dict] = None

    @property
    def allowed_citation_ids(self) -> set[str]:
        """The ids a composer may cite — strictly the grounded evidence."""
        return {item.citation_id for item in self.evidence}

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "mode": self.mode.value,
            "memory_used": self.memory_used,
            "knowledge_used": self.knowledge_used,
            "model_prior_used": self.model_prior_used,
            "informational_only": self.informational_only,
            "refused": self.refused,
            "route": self.route,
            "evidence": [e.to_dict() for e in self.evidence],
            "conflict_context": [e.to_dict() for e in self.conflict_context],
            "cautions": list(self.cautions),
            "historical_note": self.historical_note,
            "allowed_citation_ids": sorted(self.allowed_citation_ids),
        }


@dataclass(frozen=True)
class AnswerSpan:
    """One emitted span of an answer, bound to a single allowed evidence item.

    ``text`` must be a verbatim substring of the cited evidence item's text.
    This is what makes an extractive answer auditable span-by-span: the
    AnswerGuard can confirm every span is supported by the source it cites,
    forbidding any unsupported bridging text.
    """

    text: str
    citation_id: str
    source_name: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "citation_id": self.citation_id,
            "source_name": self.source_name,
        }


# Structural section kinds for a consultant report (v2.7). The kind is the
# *source of truth* for how a section is governed — never inferred by parsing
# the rendered prose. A "factual" section carries citation-bound spans (verbatim
# substrings of allowed evidence); a "judgement" section carries labelled,
# uncited analysis.
SECTION_FACTUAL = "factual"
SECTION_JUDGEMENT = "judgement"

# Every judgement block is prefixed with this exact label so it can never be
# read as a grounded finding. The AnswerGuard requires the label and forbids any
# citation marker inside a judgement section.
JUDGEMENT_LABEL = "[JUDGEMENT — not grounded in evidence]"


@dataclass(frozen=True)
class ReportSection:
    """One section of a consultant report, typed by ``kind``.

    A ``SECTION_FACTUAL`` section carries ``spans`` only: each span is a verbatim
    substring of the evidence item it cites (the v2.6 invariant). A
    ``SECTION_JUDGEMENT`` section carries ``judgement_text`` only: labelled,
    uncited prose. ``kind`` is structural, not derived from the text — the guard
    reads it from here, never by parsing the rendered report.
    """

    title: str
    kind: str  # SECTION_FACTUAL | SECTION_JUDGEMENT
    spans: List[AnswerSpan] = field(default_factory=list)
    judgement_text: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "kind": self.kind,
            "spans": [s.to_dict() for s in self.spans],
            "judgement_text": self.judgement_text,
        }


@dataclass(frozen=True)
class ReportStructure:
    """The structured carrier for a consultant report.

    Holds the ordered, typed sections. This is the source of truth the guard and
    the value-sprint metrics read; the rendered ``ComposedAnswer.text`` is a
    derived view of it, never the other way round.
    """

    sections: List[ReportSection] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"sections": [s.to_dict() for s in self.sections]}


@dataclass
class ComposedAnswer:
    """A rendered answer. Structural fields mirror the package, never the SLM."""

    text: str
    mode: ComposerMode
    citations: List[str]
    composer_backend: str
    model_prior_labelled: bool
    informational_only: bool
    refused: bool
    fell_back: bool = False
    # Optional per-span citation binding. Empty for whole-chunk composers (the
    # template and SLM composers); populated by the extractive composer so each
    # bullet maps to exactly one allowed evidence item.
    spans: List[AnswerSpan] = field(default_factory=list)
    # Optional structured consultant report (v2.7). ``None`` for every composer
    # except the consultant report composer, so all report-aware guard checks
    # and metrics are no-ops on ordinary answers.
    report: Optional["ReportStructure"] = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "mode": self.mode.value,
            "citations": list(self.citations),
            "composer_backend": self.composer_backend,
            "model_prior_labelled": self.model_prior_labelled,
            "informational_only": self.informational_only,
            "refused": self.refused,
            "fell_back": self.fell_back,
            "spans": [s.to_dict() for s in self.spans],
            "report": self.report.to_dict() if self.report else None,
        }


def _append_cautions(body: str, package: GroundingPackage) -> str:
    """Append the package's cautions and historical note to a rendered body.

    Shared by every deterministic composer so a stale-source caution or a
    superseded-memory note is surfaced identically however the body was built.
    """
    cautions = [f"! {c}" for c in package.cautions]
    if package.historical_note:
        cautions.append(f"! {package.historical_note}")
    if cautions:
        body = body + "\n" + "\n".join(cautions)
    return body


def _citations_for(package: GroundingPackage) -> List[str]:
    """Ids that may be cited in the final answer for this package's mode.

    Only grounded and pack-summary answers carry citations. Refusals,
    conflict explanations and model-prior answers cite nothing — there is no
    grounded source behind them.
    """
    if package.mode in (ComposerMode.GROUNDED, ComposerMode.PACK_SUMMARY):
        return sorted(package.allowed_citation_ids)
    return []


class AssistantComposer(ABC):
    """Interface for anything that renders a :class:`GroundingPackage`."""

    name: str = "composer"

    @abstractmethod
    def compose(self, package: GroundingPackage) -> ComposedAnswer:
        """Render ``package`` into a :class:`ComposedAnswer`."""
        raise NotImplementedError


class TemplateComposer(AssistantComposer):
    """The default deterministic composer. No model, fully offline."""

    name = "template"

    def compose(self, package: GroundingPackage) -> ComposedAnswer:
        text = self._render(package)
        return ComposedAnswer(
            text=text,
            mode=package.mode,
            citations=_citations_for(package),
            composer_backend=self.name,
            model_prior_labelled=(
                package.mode == ComposerMode.MODEL_PRIOR_LABELLED),
            informational_only=package.informational_only,
            refused=package.refused,
            fell_back=False,
        )

    # -- rendering --

    def _render(self, package: GroundingPackage) -> str:
        if package.mode == ComposerMode.GROUNDED:
            body = self._render_grounded(package)
        elif package.mode == ComposerMode.CONFLICT_EXPLANATION:
            body = self._render_conflict(package)
        elif package.mode == ComposerMode.MODEL_PRIOR_LABELLED:
            body = self._render_model_prior(package)
        elif package.mode == ComposerMode.PACK_SUMMARY:
            body = self._render_pack_summary(package)
        else:
            body = self._render_refusal(package)
        return _append_cautions(body, package)

    def _render_grounded(self, package: GroundingPackage) -> str:
        lead = "Based on grounded evidence:"
        if package.informational_only:
            lead = "Informational only (not professional advice). " + lead
        lines = [lead]
        for item in package.evidence:
            tag = (f" [{item.source_name}]" if item.source_name else "")
            lines.append(f"  - {item.text}{tag} [{item.citation_id}]")
        return "\n".join(lines)

    def _render_conflict(self, package: GroundingPackage) -> str:
        lines = [
            "No grounded answer: the closest stored memory is a near-miss the "
            "verifier rejected, so it cannot be cited as the answer.",
        ]
        for item in package.conflict_context:
            lines.append(f"  (rejected near-miss: {item.text})")
        return "\n".join(lines)

    def _render_model_prior(self, package: GroundingPackage) -> str:
        return (
            "[Unverified model prior] No project memory or imported knowledge "
            "grounds this. The following is the model's own prior and is not "
            "backed by a trusted source; verify before relying on it.")

    def _render_pack_summary(self, package: GroundingPackage) -> str:
        lines = ["Pack knowledge summary:"]
        for item in package.evidence:
            name = item.source_name or item.citation_id
            lines.append(f"  - {name} [{item.citation_id}]")
        if not package.evidence:
            lines.append("  (no imported knowledge sources)")
        return "\n".join(lines)

    def _render_refusal(self, package: GroundingPackage) -> str:
        return (
            "No grounded memory or imported knowledge supports an answer, so I "
            "will not answer. Add a memory or import a source, then ask again.")


class LocalSLMComposer(AssistantComposer):
    """Opt-in composer that uses a local SLM to smooth the prose.

    The SLM only rewrites the *body text*. Every decision and every citable id
    is fixed by the package. Any attempt to invent a citation, cite on a
    non-grounded answer, or return malformed output falls back to the
    deterministic template — which is always safe.
    """

    def __init__(self, backend: LocalSLMBackend,
                 fallback: Optional[TemplateComposer] = None):
        self.backend = backend
        self._fallback = fallback or TemplateComposer()
        self.name = f"slm:{getattr(backend, 'name', 'unknown')}"

    def compose(self, package: GroundingPackage) -> ComposedAnswer:
        if not self.backend.is_available():
            return self._fall_back(package)
        try:
            raw = self.backend.generate(self._build_prompt(package))
        except Exception:
            return self._fall_back(package)
        text = self._validate(raw, package)
        if text is None:
            return self._fall_back(package)
        return ComposedAnswer(
            text=text,
            mode=package.mode,
            citations=_citations_for(package),
            composer_backend=self.name,
            model_prior_labelled=(
                package.mode == ComposerMode.MODEL_PRIOR_LABELLED),
            informational_only=package.informational_only,
            refused=package.refused,
            fell_back=False,
        )

    # -- internals --

    def _fall_back(self, package: GroundingPackage) -> ComposedAnswer:
        answer = self._fallback.compose(package)
        answer.composer_backend = self.name
        answer.fell_back = True
        return answer

    def _validate(self, raw: str, package: GroundingPackage) -> Optional[str]:
        """Return safe text, or ``None`` to force a fallback.

        Rejects empty output, invented citations, and any citation at all on a
        non-grounded answer.
        """
        if raw is None:
            return None
        text = raw.strip()
        if not text:
            return None
        markers = _CITATION_RE.findall(text)
        cited = {f"{kind}:{ident.strip()}" for kind, ident in markers}
        if cited - package.allowed_citation_ids:
            return None  # invented a citation not in the grounded evidence
        if cited and package.mode not in (
                ComposerMode.GROUNDED, ComposerMode.PACK_SUMMARY):
            return None  # cited a source on a refused / ungrounded answer
        return text

    def _build_prompt(self, package: GroundingPackage) -> str:
        """Build a deterministic, evidence-only prompt for the SLM.

        The SLM is given the allowed evidence and citation ids and is told it
        may not invent citations. This is advisory; the hard guarantee is the
        validation above, not the prompt.
        """
        lines = [
            "You are a careful assistant. Compose a short answer using ONLY the "
            "evidence below. You may cite an evidence item with its id in "
            "square brackets, but you must not invent any id.",
            f"Mode: {package.mode.value}",
            f"Question: {package.query}",
        ]
        if package.evidence:
            lines.append("Evidence:")
            for item in package.evidence:
                lines.append(f"  [{item.citation_id}] {item.text}")
        else:
            lines.append("Evidence: (none — you must refuse to answer)")
        if package.cautions:
            lines.append("Cautions to preserve: " + "; ".join(package.cautions))
        return "\n".join(lines)


# Sentence boundary: whitespace that follows sentence-ending punctuation. The
# split consumes only the separating whitespace, so every resulting piece is a
# contiguous substring of the original text — the verbatim guarantee the
# extractive composer and AnswerGuard rely on.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> List[str]:
    """Split ``text`` into trimmed sentence spans, each a verbatim substring.

    ``str.strip`` only removes leading/trailing whitespace, so each returned
    span is still a contiguous substring of ``text`` (``span in text`` holds).
    """
    return [p.strip() for p in _SENTENCE_SPLIT_RE.split(text or "") if p.strip()]


class ExtractiveMultiChunkComposer(AssistantComposer):
    """Opt-in composer that quotes the most relevant span from each source.

    Where :class:`TemplateComposer` echoes each evidence item *whole*, this
    composer selects, from every allowed evidence item, the sentence(s) whose
    lexical overlap with the query is highest, and renders them as per-span,
    citation-bound bullets. It is strictly **extractive**:

    - every emitted span is a *verbatim substring* of the evidence item it
      cites — no paraphrase, no token added, no two chunks merged;
    - the set of citable ids is unchanged (still the package's grounded
      evidence), so it can never cite more sources than the template would;
    - only ``GROUNDED`` answers are reshaped. Refusal, conflict, model-prior
      and pack-summary answers are delegated to the template verbatim, so every
      non-grounded path is byte-identical to the default.

    The span scoring reuses the frozen deterministic ``content_tokens`` /
    ``overlap_coefficient`` primitives from the evidence ranker, so it adds no
    new notion of relevance. The :class:`~slm.answer_guard.AnswerGuard` enforces
    the verbatim-substring invariant after the fact; a span that is not
    supported by its cited evidence is rejected and recomposed by the template.
    """

    name = "extractive"

    def __init__(self, *, max_spans_per_item: int = 2,
                 fallback: Optional[TemplateComposer] = None):
        if max_spans_per_item < 1:
            raise ValueError("max_spans_per_item must be >= 1")
        self.max_spans_per_item = max_spans_per_item
        self._template = fallback or TemplateComposer()

    def compose(self, package: GroundingPackage) -> ComposedAnswer:
        if package.mode != ComposerMode.GROUNDED:
            # Non-grounded modes are unchanged: render with the template, but
            # record that the extractive composer was the one selected.
            answer = self._template.compose(package)
            answer.composer_backend = self.name
            return answer
        spans = self._select_spans(package)
        text = _append_cautions(self._render_spans(package, spans), package)
        return ComposedAnswer(
            text=text,
            mode=package.mode,
            citations=_citations_for(package),
            composer_backend=self.name,
            model_prior_labelled=False,
            informational_only=package.informational_only,
            refused=package.refused,
            fell_back=False,
            spans=spans,
        )

    # -- internals --

    def _select_spans(self, package: GroundingPackage) -> List[AnswerSpan]:
        # Lazy import: the evidence ranker imports EvidenceItem from this module,
        # so a top-level import would be circular.
        from retrieval.evidence_ranker import content_tokens, overlap_coefficient

        query_tokens = set(content_tokens(package.query))
        spans: List[AnswerSpan] = []
        for item in package.evidence:
            sentences = _split_sentences(item.text)
            if not sentences:
                continue
            scored = sorted(
                ((overlap_coefficient(query_tokens,
                                      set(content_tokens(sentence))), idx, sentence)
                 for idx, sentence in enumerate(sentences)),
                key=lambda triple: (triple[0], -triple[1]),
                reverse=True,
            )
            chosen = [(idx, sentence) for score, idx, sentence in scored
                      if score > 0][: self.max_spans_per_item]
            if not chosen:
                # No lexical overlap with any sentence: keep the leading
                # sentence so a cited source is never silently dropped.
                chosen = [(0, sentences[0])]
            chosen.sort(key=lambda pair: pair[0])  # restore reading order
            for _, sentence in chosen:
                spans.append(AnswerSpan(
                    text=sentence,
                    citation_id=item.citation_id,
                    source_name=item.source_name,
                ))
        return spans

    def _render_spans(self, package: GroundingPackage,
                      spans: List[AnswerSpan]) -> str:
        lead = "Based on grounded evidence:"
        if package.informational_only:
            lead = "Informational only (not professional advice). " + lead
        lines = [lead]
        for span in spans:
            tag = f" [{span.source_name}]" if span.source_name else ""
            lines.append(f"  - {span.text}{tag} [{span.citation_id}]")
        return "\n".join(lines)


class ConsultantReportComposer(AssistantComposer):
    """Opt-in composer that structures grounded evidence as a consultant report.

    This is **not** an abstractive report writer. It is a deterministic
    *structuring* tool with a hard partition between two section kinds:

    - **factual** sections (Executive summary, Current state, Evidence) are built
      from citation-bound, verbatim spans — delegated to
      :class:`ExtractiveMultiChunkComposer`, so every claim is a substring of the
      evidence it cites and is guard-enforced exactly as in v2.6;
    - **judgement** sections (Risks, Options, Recommendation, Assumptions, Open
      questions, Next actions) carry labelled, **uncited** analysis. With no
      model in the loop the composer never fabricates a conclusion: each
      judgement block is either mechanically derived from the package
      (``cautions`` / ``conflict_context``) or degrades to a labelled
      placeholder. An unsupported recommendation is labelled and uncited, never
      presented as a grounded finding.

    The :class:`ReportStructure` carrier on the answer is the source of truth for
    section kind; the rendered prose is a derived view of it. The
    :class:`~slm.answer_guard.AnswerGuard` reads that carrier — it never infers a
    section's kind from the text. As with every composer, only ``GROUNDED``
    answers are reshaped; refusal, conflict, model-prior and pack-summary answers
    are delegated to the template byte-for-byte.
    """

    name = "consultant"

    def __init__(self, *, fallback: Optional[TemplateComposer] = None,
                 extractive: Optional["ExtractiveMultiChunkComposer"] = None):
        self._template = fallback or TemplateComposer()
        self._extractive = extractive or ExtractiveMultiChunkComposer(
            fallback=self._template)

    def compose(self, package: GroundingPackage) -> ComposedAnswer:
        if package.mode != ComposerMode.GROUNDED:
            # Non-grounded modes are byte-identical to the template; only the
            # recorded backend name changes.
            answer = self._template.compose(package)
            answer.composer_backend = self.name
            return answer
        sections = self._build_sections(package)
        spans = [span for section in sections
                 if section.kind == SECTION_FACTUAL for span in section.spans]
        body = self._render_report(package, sections)
        text = _append_cautions(body, package)
        return ComposedAnswer(
            text=text,
            mode=package.mode,
            citations=_citations_for(package),
            composer_backend=self.name,
            model_prior_labelled=False,
            informational_only=package.informational_only,
            refused=package.refused,
            fell_back=False,
            spans=spans,
            report=ReportStructure(sections=sections),
        )

    # -- section construction --

    def _build_sections(self, package: GroundingPackage) -> List[ReportSection]:
        relevant_spans = self._extractive._select_spans(package)
        evidence_spans = self._evidence_spans(package)
        return [
            ReportSection(title="Executive summary", kind=SECTION_FACTUAL,
                          spans=relevant_spans[:1]),
            ReportSection(title="Current state", kind=SECTION_FACTUAL,
                          spans=relevant_spans),
            ReportSection(title="Evidence", kind=SECTION_FACTUAL,
                          spans=evidence_spans),
            ReportSection(title="Risks", kind=SECTION_JUDGEMENT,
                          judgement_text=self._risks_text(package)),
            ReportSection(title="Options", kind=SECTION_JUDGEMENT,
                          judgement_text=self._placeholder("Options")),
            ReportSection(title="Recommendation", kind=SECTION_JUDGEMENT,
                          judgement_text=self._recommendation_text(package)),
            ReportSection(title="Assumptions", kind=SECTION_JUDGEMENT,
                          judgement_text=self._placeholder("Assumptions")),
            ReportSection(title="Open questions", kind=SECTION_JUDGEMENT,
                          judgement_text=self._open_questions_text(package)),
            ReportSection(title="Next actions", kind=SECTION_JUDGEMENT,
                          judgement_text=self._placeholder("Next actions")),
        ]

    def _evidence_spans(self, package: GroundingPackage) -> List[AnswerSpan]:
        """One leading-sentence span per evidence item, so every allowed source
        appears at least once. Each span is a verbatim substring of its source.
        """
        spans: List[AnswerSpan] = []
        for item in package.evidence:
            sentences = _split_sentences(item.text)
            lead = sentences[0] if sentences else item.text
            spans.append(AnswerSpan(
                text=lead,
                citation_id=item.citation_id,
                source_name=item.source_name,
            ))
        return spans

    # -- judgement construction (labelled, uncited, never fabricated) --

    def _label(self, body: str) -> str:
        """Prefix a judgement body with the mandatory label and strip any
        citation marker, so a judgement block can never read as grounded.
        """
        return f"{JUDGEMENT_LABEL}\n{_CITATION_RE.sub('', body).strip()}"

    def _placeholder(self, what: str) -> str:
        return self._label(
            f"{what} require author judgement; none can be derived from the "
            "allowed evidence alone.")

    def _risks_text(self, package: GroundingPackage) -> str:
        notes: List[str] = [f"- {c}" for c in package.cautions]
        if package.historical_note:
            notes.append(f"- {package.historical_note}")
        for item in package.conflict_context:
            notes.append(
                "- A near-miss was rejected by the verifier and is not "
                f"grounded: {item.text}")
        body = "\n".join(notes) if notes else (
            "No evidence-derived risks identified; author judgement required.")
        return self._label(body)

    def _recommendation_text(self, package: GroundingPackage) -> str:
        # Deterministic and model-free: never fabricate a recommendation. Degrade
        # to a labelled placeholder; the grounded basis to act on lives in the
        # factual sections above.
        return self._label(
            "No recommendation can be derived from the allowed evidence; author "
            "judgement required. The factual sections above are evidence-bound "
            "and may inform that judgement.")

    def _open_questions_text(self, package: GroundingPackage) -> str:
        notes = [
            f"- What resolves the rejected near-miss: {item.text}"
            for item in package.conflict_context
        ]
        body = "\n".join(notes) if notes else (
            "No open questions derived from evidence; author judgement "
            "required.")
        return self._label(body)

    # -- rendering --

    def _render_report(self, package: GroundingPackage,
                       sections: List[ReportSection]) -> str:
        lead = ("Consultant report (deterministic; factual sections are "
                "evidence-bound, judgement sections are labelled and uncited):")
        if package.informational_only:
            lead = "Informational only (not professional advice). " + lead
        lines = [lead]
        for section in sections:
            lines.append("")
            lines.append(f"## {section.title}")
            if section.kind == SECTION_FACTUAL:
                if not section.spans:
                    lines.append("  (no grounded evidence for this section)")
                for span in section.spans:
                    tag = f" [{span.source_name}]" if span.source_name else ""
                    lines.append(f"  - {span.text}{tag} [{span.citation_id}]")
            else:
                lines.append(section.judgement_text or "")
        return "\n".join(lines)
