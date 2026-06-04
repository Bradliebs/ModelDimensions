"""v6.3 Hugging Face Metadata Adapter — metadata inspection only.

This module converts Hugging Face-style **dataset card metadata** into an
:class:`agent.data_intake.ExternalDatasetCandidate` and runs it through the
existing v6.2 Clean Data Intake Lane. It is a thin, deterministic adapter: it
maps declared metadata fields and applies conservative risk inference, then
delegates the actual decision to :func:`agent.data_intake.assess_candidate`, so
the intake lane remains the single source of truth.

This is **metadata inspection only**. It never:

* downloads full datasets or streams dataset rows,
* writes to the MemoryLedger,
* writes to the source registry,
* creates or applies source/memory proposals,
* changes retrieval, ranking, grounding, or chat routing,
* marks any Hugging Face dataset as trusted knowledge automatically.

It mutates durable project state only when the caller passes an explicit output
path to :func:`agent.data_intake.write_assessment_report`, and even then writes
*only* an assessment report. The module imports no memory/source/proposal writer
and no dataset-download client, so it cannot do any of those even by accident.

Core invariants carried forward from v6.2:

* External data must not become trusted knowledge automatically.
* ``approved_for_eval`` is a distinct, weaker tier than
  ``approved_for_knowledge``.
* A missing licence, an unknown provenance, possible/declared PII, and
  non-commercial or research-only licences can never auto-classify as
  ``approved_for_knowledge``.
* Gated and private datasets cannot be openly verified, so they are routed to
  human review rather than trusted automatically.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from agent.data_intake import (
    DataIntakeAssessment,
    DataIntakeDecision,
    ExternalDatasetCandidate,
    IntakeLane,
    assess_candidate,
    render_intake_assessment_markdown,
)

# --------------------------------------------------------------------------- #
# Inference vocabularies (deterministic; no network, no download).
# --------------------------------------------------------------------------- #

# A Hugging Face ``gated`` flag may be False, ``"auto"``, or ``"manual"``.
_GATED_TRUTHY = frozenset({"true", "auto", "manual", "yes", "1", "gated"})

_PERSONAL_DATA_TAG_MARKERS: Tuple[str, ...] = (
    "pii", "personal-data", "personal_data", "personally-identifiable",
    "personal-information", "personal_information", "contains-pii",
    "email-addresses", "phone-numbers", "ssn",
)
_SENSITIVE_DOMAIN_MARKERS: Tuple[str, ...] = (
    "medical", "clinical", "healthcare", "patient", "biomedical",
    "legal", "court", "litigation", "finance", "financial", "banking",
)
_SYNTHETIC_TAG_MARKERS: Tuple[str, ...] = (
    "synthetic", "synthetic-data", "machine-generated", "llm-generated",
    "ai-generated",
)
_BENCHMARK_TAG_MARKERS: Tuple[str, ...] = (
    "benchmark", "leaderboard", "test-set",
)

# Tag prefixes Hugging Face uses to encode structured card metadata.
_TAG_PREFIXES = ("license:", "language:", "task_categories:", "size_categories:")


def _norm(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def _first(values: Sequence[str]) -> str:
    for value in values:
        text = str(value).strip()
        if text:
            return text
    return ""


def _as_tuple(value) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    return tuple(str(v).strip() for v in value if str(v).strip())


def _is_gated(value) -> bool:
    if isinstance(value, bool):
        return value
    return _norm(str(value)) in _GATED_TRUTHY


# --------------------------------------------------------------------------- #
# Frozen data structures.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HuggingFaceDatasetMetadata:
    """Declared Hugging Face-style dataset card metadata (no content).

    This holds only what a dataset *card* declares about itself. No dataset rows
    are referenced or downloaded.
    """

    dataset_id: str
    license: Optional[str] = None
    tags: Tuple[str, ...] = field(default_factory=tuple)
    language: Tuple[str, ...] = field(default_factory=tuple)
    task_categories: Tuple[str, ...] = field(default_factory=tuple)
    size_categories: Tuple[str, ...] = field(default_factory=tuple)
    pretty_name: str = ""
    description: str = ""
    homepage: str = ""
    citation: str = ""
    created_at: Optional[str] = None
    last_modified: Optional[str] = None
    gated: bool = False
    private: bool = False
    downloads: int = 0
    likes: int = 0
    dataset_info: Optional[dict] = None
    contains_personal_data: Optional[bool] = None

    @classmethod
    def from_dict(cls, payload: dict) -> "HuggingFaceDatasetMetadata":
        if not isinstance(payload, dict):
            raise ValueError("Hugging Face metadata record must be a JSON object")
        card = payload.get("card_data") or payload.get("cardData") or {}
        if not isinstance(card, dict):
            card = {}
        tags = _as_tuple(payload.get("tags") or card.get("tags"))

        dataset_id = payload.get("dataset_id") or payload.get("id") or payload.get("name")
        if not dataset_id or not str(dataset_id).strip():
            raise ValueError("Hugging Face metadata is missing a non-empty 'id'/'dataset_id'")

        licence = (
            card.get("license") or card.get("licence")
            or payload.get("license") or payload.get("licence")
            or _value_from_tags(tags, "license:")
        )
        language = _as_tuple(
            card.get("language") or payload.get("language")
            or _values_from_tags(tags, "language:"))
        task_categories = _as_tuple(
            card.get("task_categories") or payload.get("task_categories")
            or _values_from_tags(tags, "task_categories:"))
        size_categories = _as_tuple(
            card.get("size_categories") or payload.get("size_categories")
            or _values_from_tags(tags, "size_categories:"))
        pretty_name = str(
            card.get("pretty_name") or payload.get("pretty_name") or "").strip()

        pii = payload.get("contains_personal_data")
        if pii is None:
            pii = card.get("contains_personal_data")
        if isinstance(pii, str):
            text = pii.strip().lower()
            if text in {"true", "yes", "present", "1"}:
                pii = True
            elif text in {"false", "no", "absent", "none", "0"}:
                pii = False
            else:
                pii = None

        return cls(
            dataset_id=str(dataset_id),
            license=str(licence).strip() if licence else None,
            tags=tags,
            language=language,
            task_categories=task_categories,
            size_categories=size_categories,
            pretty_name=pretty_name,
            description=str(payload.get("description") or card.get("description") or "").strip(),
            homepage=str(payload.get("homepage") or card.get("homepage") or "").strip(),
            citation=str(payload.get("citation") or card.get("citation") or "").strip(),
            created_at=payload.get("created_at") or payload.get("createdAt"),
            last_modified=payload.get("last_modified") or payload.get("lastModified"),
            gated=_is_gated(payload.get("gated", False)),
            private=bool(payload.get("private", False)),
            downloads=int(payload.get("downloads") or 0),
            likes=int(payload.get("likes") or 0),
            dataset_info=payload.get("dataset_info") if isinstance(
                payload.get("dataset_info"), dict) else None,
            contains_personal_data=pii if isinstance(pii, bool) else None,
        )

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "license": self.license,
            "tags": list(self.tags),
            "language": list(self.language),
            "task_categories": list(self.task_categories),
            "size_categories": list(self.size_categories),
            "pretty_name": self.pretty_name,
            "description": self.description,
            "homepage": self.homepage,
            "citation": self.citation,
            "created_at": self.created_at,
            "last_modified": self.last_modified,
            "gated": self.gated,
            "private": self.private,
            "downloads": self.downloads,
            "likes": self.likes,
            "contains_personal_data": self.contains_personal_data,
        }


@dataclass(frozen=True)
class HuggingFaceAdapterResult:
    """The candidate + intake assessment derived from one metadata record.

    ``adapter_notes`` are informational only; the decision comes entirely from
    the v6.2 intake lane (``assessment``).
    """

    metadata: HuggingFaceDatasetMetadata
    candidate: ExternalDatasetCandidate
    assessment: DataIntakeAssessment
    adapter_notes: Tuple[str, ...] = ()

    @property
    def decision(self) -> DataIntakeDecision:
        return self.assessment.decision

    @property
    def lane(self) -> IntakeLane:
        return self.assessment.lane

    @property
    def approved_for_eval(self) -> bool:
        return self.assessment.approved_for_eval

    @property
    def approved_for_knowledge(self) -> bool:
        return self.assessment.approved_for_knowledge

    @property
    def permits_eval_use(self) -> bool:
        return self.assessment.permits_eval_use

    def to_dict(self) -> dict:
        return {
            "_record": "hf_metadata_adapter_result",
            "metadata": self.metadata.to_dict(),
            "adapter_notes": list(self.adapter_notes),
            "assessment": self.assessment.to_dict(),
        }


# --------------------------------------------------------------------------- #
# Tag helpers.
# --------------------------------------------------------------------------- #


def _values_from_tags(tags: Sequence[str], prefix: str) -> Tuple[str, ...]:
    out: List[str] = []
    for tag in tags:
        text = str(tag).strip()
        if text.lower().startswith(prefix):
            value = text[len(prefix):].strip()
            if value:
                out.append(value)
    return tuple(out)


def _value_from_tags(tags: Sequence[str], prefix: str) -> Optional[str]:
    values = _values_from_tags(tags, prefix)
    return values[0] if values else None


def _plain_tag_text(metadata: HuggingFaceDatasetMetadata) -> Tuple[str, ...]:
    """All free-form tag/category text used for risk inference (prefixes stripped)."""
    out: List[str] = []
    for tag in metadata.tags:
        text = str(tag).strip()
        lowered = text.lower()
        if any(lowered.startswith(p) for p in _TAG_PREFIXES):
            continue
        if text:
            out.append(lowered)
    out.extend(c.lower() for c in metadata.task_categories)
    return tuple(out)


def _has_any(haystack: Sequence[str], markers: Sequence[str]) -> bool:
    return any(marker in item for item in haystack for marker in markers)


def _is_synthetic(metadata: HuggingFaceDatasetMetadata) -> bool:
    return _has_any(_plain_tag_text(metadata), _SYNTHETIC_TAG_MARKERS)


def _is_benchmark(metadata: HuggingFaceDatasetMetadata) -> bool:
    return _has_any(_plain_tag_text(metadata), _BENCHMARK_TAG_MARKERS)


def _has_personal_data_signal(metadata: HuggingFaceDatasetMetadata) -> bool:
    tags = _plain_tag_text(metadata)
    return _has_any(tags, _PERSONAL_DATA_TAG_MARKERS) or _has_any(
        tags, _SENSITIVE_DOMAIN_MARKERS)


# --------------------------------------------------------------------------- #
# Deterministic field mapping + conservative risk inference.
# --------------------------------------------------------------------------- #


def _source_type(metadata: HuggingFaceDatasetMetadata) -> str:
    if _is_benchmark(metadata):
        return "benchmark"
    if _is_synthetic(metadata):
        return "synthetic"
    return "huggingface_dataset"


def _intended_use(metadata: HuggingFaceDatasetMetadata) -> str:
    if _is_benchmark(metadata):
        return "benchmark"
    if _is_synthetic(metadata):
        return ""  # the synthetic flag itself routes this to the eval tier
    return "knowledge_base"


def _provenance(metadata: HuggingFaceDatasetMetadata) -> str:
    # Gated/private datasets cannot be openly verified -> treat origin as
    # unestablished so the intake lane routes them to human review.
    if metadata.gated or metadata.private:
        return "unknown"
    signals: List[str] = []
    if metadata.citation:
        signals.append("cited")
    if metadata.homepage:
        signals.append("homepage")
    if signals:
        return f"huggingface:{metadata.dataset_id} ({', '.join(signals)})"
    return "unknown"


def _personal_data(metadata: HuggingFaceDatasetMetadata) -> Optional[bool]:
    # An explicit declaration always wins.
    if metadata.contains_personal_data is not None:
        return metadata.contains_personal_data
    # A personal-data tag or a sensitive personal domain -> possible PII.
    if _has_personal_data_signal(metadata):
        return None
    # Otherwise PII status is undeclared -> conservatively undeclared (review).
    return None


def _size_hint(metadata: HuggingFaceDatasetMetadata) -> Optional[str]:
    if metadata.size_categories:
        return _first(metadata.size_categories)
    info = metadata.dataset_info or {}
    examples = info.get("num_examples") or info.get("num_rows")
    if examples:
        return f"{examples} examples"
    return None


def _publisher(metadata: HuggingFaceDatasetMetadata) -> str:
    if "/" in metadata.dataset_id:
        return metadata.dataset_id.split("/", 1)[0]
    return ""


def hf_metadata_to_candidate(
    metadata: HuggingFaceDatasetMetadata,
) -> ExternalDatasetCandidate:
    """Pure, deterministic conversion of HF metadata to an intake candidate.

    Reads only declared metadata; downloads nothing and writes nothing.
    """
    source_url = metadata.homepage or f"https://huggingface.co/datasets/{metadata.dataset_id}"
    title = metadata.pretty_name or metadata.dataset_id
    notes = "; ".join(
        n for n in (
            "gated dataset" if metadata.gated else "",
            "private dataset" if metadata.private else "",
        ) if n
    )
    return ExternalDatasetCandidate(
        dataset_id=metadata.dataset_id,
        source_url=source_url,
        source_type=_source_type(metadata),
        title=title,
        description=metadata.description,
        licence=metadata.license,
        provenance=_provenance(metadata),
        publisher=_publisher(metadata),
        language=", ".join(metadata.language),
        task_categories=metadata.task_categories,
        size_hint=_size_hint(metadata),
        last_updated=metadata.last_modified or metadata.created_at,
        intended_use=_intended_use(metadata),
        contains_personal_data=_personal_data(metadata),
        is_synthetic=_is_synthetic(metadata),
        sample_available=None,
        notes=notes,
    )


def _adapter_notes(metadata: HuggingFaceDatasetMetadata) -> Tuple[str, ...]:
    notes: List[str] = []
    if metadata.gated:
        notes.append("gated dataset: access-restricted, so provenance is treated "
                     "as unverified and routed to human review")
    if metadata.private:
        notes.append("private dataset: not openly verifiable; routed to human review")
    if _has_personal_data_signal(metadata) and metadata.contains_personal_data is None:
        notes.append("tags suggest possible personal data; PII held for review")
    if _is_synthetic(metadata):
        notes.append("synthetic dataset: eligible for the eval tier only, never "
                     "trusted knowledge automatically")
    if _is_benchmark(metadata):
        notes.append("benchmark dataset: eligible for the eval tier only, never "
                     "trusted knowledge automatically")
    return tuple(notes)


# --------------------------------------------------------------------------- #
# Public assessment entry points.
# --------------------------------------------------------------------------- #


def assess_hf_metadata(
    metadata: HuggingFaceDatasetMetadata,
    *,
    now: Optional[datetime] = None,
) -> HuggingFaceAdapterResult:
    """Convert one HF metadata record and assess it via the v6.2 intake lane."""
    candidate = hf_metadata_to_candidate(metadata)
    assessment = assess_candidate(candidate, now=now)
    return HuggingFaceAdapterResult(
        metadata=metadata,
        candidate=candidate,
        assessment=assessment,
        adapter_notes=_adapter_notes(metadata),
    )


def assess_hf_metadata_records(
    records: Sequence[HuggingFaceDatasetMetadata],
    *,
    now: Optional[datetime] = None,
) -> List[HuggingFaceAdapterResult]:
    return [assess_hf_metadata(record, now=now) for record in records]


def load_hf_metadata(path) -> List[HuggingFaceDatasetMetadata]:
    """Read HF metadata records from a local JSON or JSONL file. Reads only."""
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    payloads: List[dict] = []
    if source.suffix.lower() == ".jsonl":
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            payloads.append(json.loads(stripped))
    else:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            payloads.extend(parsed)
        else:
            payloads.append(parsed)
    return [HuggingFaceDatasetMetadata.from_dict(p) for p in payloads]


# --------------------------------------------------------------------------- #
# Rendering (no durable writes).
# --------------------------------------------------------------------------- #


def render_hf_adapter_markdown(result: HuggingFaceAdapterResult) -> str:
    """Render a deterministic report for one HF metadata assessment."""
    meta = result.metadata
    lines = [
        "# Hugging Face metadata assessment (v6.3; metadata-only)",
        "",
        f"- Hugging Face dataset: {meta.dataset_id}",
        f"- Gated: {str(meta.gated).lower()}",
        f"- Private: {str(meta.private).lower()}",
        "",
    ]
    if result.adapter_notes:
        lines.append("## Adapter notes")
        for note in result.adapter_notes:
            lines.append(f"- {note}")
        lines.append("")
    lines.append(render_intake_assessment_markdown(result.assessment))
    return "\n".join(lines)


def render_hf_adapter_summary_markdown(
    results: Sequence[HuggingFaceAdapterResult],
) -> str:
    """Render a one-line-per-dataset summary across many HF assessments."""
    lines = ["# Hugging Face metadata summary (v6.3; metadata-only)", ""]
    for result in results:
        lines.append(
            f"- {result.metadata.dataset_id}: {result.decision.value} "
            f"(lane: {result.lane.value})"
        )
    return "\n".join(lines)
