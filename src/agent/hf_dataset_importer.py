"""Source-governed Hugging Face dataset importer (v1.9).

This is a *small, licence-aware, opt-in* bridge from a Hugging Face dataset into
either a pack's knowledge library or a pack's evaluation samples. It exists so a
consultant can pull a handful of governed reference or evaluation rows from the
Hub without ever bulk-downloading a dataset or silently trusting unlicensed
text.

What makes it safe by construction:

* **No silent download.** Sampling reads from a local fixture, or from the Hub
  only when ``allow_network`` is *explicitly* set. With neither, sampling raises
  rather than reaching out. Tests run entirely on fixtures.
* **Small by default.** ``sample_size`` is capped at :data:`MAX_SAMPLE_SIZE`
  (100). A larger request is clamped, not honoured.
* **Streaming by default.** ``streaming=True`` so the full dataset is never
  materialised; only the capped sample is read.
* **Licence-gated.** An unknown licence is rejected for ``knowledge`` mode and
  allowed only with a warning for ``eval`` mode, per the chosen
  :class:`LicensePolicy`. ``medical``/``legal`` domains additionally require a
  non-``unknown`` authority.
* **Memory is never touched.** ``knowledge`` mode writes through the frozen
  :class:`KnowledgeLibrary` import path only; ``eval`` mode writes a samples file
  only. Neither path writes to the ``MemoryLedger``.

The geometry, grounding, lifecycle, and pack-builder semantics are unchanged:
this module only *decides whether and where* sampled rows may land.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import List, Optional

from agent.hf_metadata import HFDatasetMetadata, fetch_metadata, licence_is_known
from agent.knowledge_library import KnowledgeLibrary
from agent.knowledge_sources import KnowledgeDomain, SourceAuthority
from agent.project_packs import ProjectPack

# Hard ceiling on how many rows a single import may pull. The point of this
# importer is small governed samples, not bulk ingestion.
MAX_SAMPLE_SIZE = 100

# Domains whose content carries real-world risk: they may not be imported under
# an unknown authority.
_HIGH_RISK_DOMAINS = {KnowledgeDomain.MEDICAL, KnowledgeDomain.LEGAL}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class HFDatasetMode(str, Enum):
    """Where sampled rows are allowed to land."""

    KNOWLEDGE = "knowledge"
    EVAL = "eval"


class LicensePolicy(str, Enum):
    """How an unknown/declared-absent licence is handled."""

    REQUIRE_LICENSE = "require_license"
    ALLOW_UNKNOWN_FOR_EVAL_ONLY = "allow_unknown_for_eval_only"
    REJECT_UNKNOWN = "reject_unknown"


# ---------------------------------------------------------------------------
# Specs / samples / report
# ---------------------------------------------------------------------------

@dataclass
class HFDatasetSpec:
    """A declarative, governed request to sample a Hugging Face dataset.

    ``local_fixture`` (a JSONL file) and ``card_path`` (a JSON dataset card)
    keep the whole flow offline for tests and demos. ``allow_network`` must be
    explicitly set for any Hub access to be attempted, and even then only a
    capped sample is read.
    """

    dataset_id: str
    split: str = "train"
    text_fields: List[str] = field(default_factory=lambda: ["text"])
    sample_size: int = MAX_SAMPLE_SIZE
    domain: str = KnowledgeDomain.GENERAL.value
    authority: str = SourceAuthority.UNKNOWN.value
    mode: HFDatasetMode = HFDatasetMode.EVAL
    license_policy: LicensePolicy = LicensePolicy.ALLOW_UNKNOWN_FOR_EVAL_ONLY
    streaming: bool = True
    config_name: Optional[str] = None
    revision: Optional[str] = None
    dataset_card_required: bool = False
    # Offline plumbing (never fetches): a local rows fixture and a local card.
    local_fixture: Optional[str] = None
    card_path: Optional[str] = None
    # Explicit, opt-in network access. Default off: nothing is downloaded.
    allow_network: bool = False

    def __post_init__(self) -> None:
        self.mode = HFDatasetMode(self.mode)
        self.license_policy = LicensePolicy(self.license_policy)
        if not self.text_fields:
            self.text_fields = ["text"]

    @property
    def effective_sample_size(self) -> int:
        """The capped number of rows that may actually be read."""
        return max(0, min(int(self.sample_size), MAX_SAMPLE_SIZE))

    @property
    def domain_enum(self) -> KnowledgeDomain:
        return KnowledgeDomain(self.domain)

    @property
    def authority_enum(self) -> SourceAuthority:
        return SourceAuthority(self.authority)

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "config_name": self.config_name,
            "split": self.split,
            "text_fields": list(self.text_fields),
            "sample_size": self.sample_size,
            "effective_sample_size": self.effective_sample_size,
            "domain": self.domain,
            "authority": self.authority,
            "mode": self.mode.value,
            "license_policy": self.license_policy.value,
            "streaming": self.streaming,
            "revision": self.revision,
            "dataset_card_required": self.dataset_card_required,
            "allow_network": self.allow_network,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HFDatasetSpec":
        if "dataset_id" not in data:
            raise ValueError("HF dataset spec needs a 'dataset_id'")
        return cls(
            dataset_id=str(data["dataset_id"]),
            split=str(data.get("split", "train")),
            text_fields=_str_list(data.get("text_fields")) or ["text"],
            sample_size=int(data.get("sample_size", MAX_SAMPLE_SIZE)),
            domain=str(data.get("domain", KnowledgeDomain.GENERAL.value)),
            authority=str(data.get("authority", SourceAuthority.UNKNOWN.value)),
            mode=HFDatasetMode(data.get("mode", HFDatasetMode.EVAL.value)),
            license_policy=LicensePolicy(
                data.get("license_policy",
                         LicensePolicy.ALLOW_UNKNOWN_FOR_EVAL_ONLY.value)),
            streaming=bool(data.get("streaming", True)),
            config_name=_opt_str(data.get("config_name")),
            revision=_opt_str(data.get("revision")),
            dataset_card_required=bool(data.get("dataset_card_required", False)),
            local_fixture=_opt_str(data.get("local_fixture")),
            card_path=_opt_str(data.get("card_path")),
            allow_network=bool(data.get("allow_network", False)),
        )


@dataclass
class HFDatasetSample:
    """One sampled row, reduced to its text plus the raw fields it came from."""

    dataset_id: str
    split: str
    index: int
    text: str
    fields: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "split": self.split,
            "index": self.index,
            "text": self.text,
            "fields": dict(self.fields),
        }


@dataclass
class HFDatasetImportReport:
    """The auditable outcome of an import attempt (including a refusal)."""

    dataset_id: str
    mode: str
    domain: str
    authority: str
    requested_sample_size: int
    effective_sample_size: int
    streaming: bool
    license: Optional[str] = None
    license_known: bool = False
    card_present: bool = False
    accepted: bool = False
    rejected_reason: Optional[str] = None
    imported_count: int = 0
    knowledge_source_id: Optional[str] = None
    eval_output_path: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    retrieved_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "mode": self.mode,
            "domain": self.domain,
            "authority": self.authority,
            "requested_sample_size": self.requested_sample_size,
            "effective_sample_size": self.effective_sample_size,
            "streaming": self.streaming,
            "license": self.license,
            "license_known": self.license_known,
            "card_present": self.card_present,
            "accepted": self.accepted,
            "rejected_reason": self.rejected_reason,
            "imported_count": self.imported_count,
            "knowledge_source_id": self.knowledge_source_id,
            "eval_output_path": self.eval_output_path,
            "warnings": list(self.warnings),
            "retrieved_at": self.retrieved_at,
        }


# ---------------------------------------------------------------------------
# Inspect / sample
# ---------------------------------------------------------------------------

def inspect_dataset(spec: HFDatasetSpec) -> dict:
    """Gather a dataset's declared metadata and preview the import decision.

    Performs no import and writes nothing. Network is touched only when the spec
    explicitly opts in; otherwise this is fully offline.
    """
    metadata = _metadata_for(spec)
    decision = _gate(spec, metadata)
    return {
        "spec": spec.to_dict(),
        "metadata": metadata.to_dict(),
        "decision": {
            "would_import": decision.allowed,
            "rejected_reason": decision.reason,
            "warnings": list(decision.warnings),
        },
    }


def sample_dataset(spec: HFDatasetSpec) -> List[HFDatasetSample]:
    """Read at most ``effective_sample_size`` rows, offline-first.

    Rows come from ``local_fixture`` when set; otherwise the Hub is read only if
    ``allow_network`` is explicitly enabled. With neither, this raises rather
    than downloading anything — silent bulk import is impossible.
    """
    cap = spec.effective_sample_size
    if cap <= 0:
        return []
    rows = _load_rows(spec, cap)
    samples: List[HFDatasetSample] = []
    for index, row in enumerate(rows):
        if index >= cap:
            break
        text = _extract_text(row, spec.text_fields)
        if not text:
            continue
        samples.append(HFDatasetSample(
            dataset_id=spec.dataset_id,
            split=spec.split,
            index=index,
            text=text,
            fields={k: row.get(k) for k in spec.text_fields if k in row},
        ))
    return samples


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

def import_hf_dataset_to_pack(spec: HFDatasetSpec,
                              pack: ProjectPack) -> HFDatasetImportReport:
    """Sample a dataset into ``pack`` as knowledge or eval samples, if allowed.

    The licence/authority/card gate runs *before* any rows are read or written.
    A refused import returns a report with ``accepted=False`` and writes nothing.
    """
    metadata = _metadata_for(spec)
    report = HFDatasetImportReport(
        dataset_id=spec.dataset_id,
        mode=spec.mode.value,
        domain=spec.domain,
        authority=spec.authority,
        requested_sample_size=int(spec.sample_size),
        effective_sample_size=spec.effective_sample_size,
        streaming=spec.streaming,
        license=metadata.license,
        license_known=metadata.license_known,
        card_present=metadata.card_present,
    )
    if int(spec.sample_size) > MAX_SAMPLE_SIZE:
        report.warnings.append(
            f"requested sample_size {spec.sample_size} exceeds cap "
            f"{MAX_SAMPLE_SIZE}; clamped to {spec.effective_sample_size}")

    decision = _gate(spec, metadata)
    report.warnings.extend(decision.warnings)
    if not decision.allowed:
        report.accepted = False
        report.rejected_reason = decision.reason
        return report

    samples = sample_dataset(spec)
    if spec.mode is HFDatasetMode.KNOWLEDGE:
        source_id = _write_knowledge(spec, pack, samples, metadata)
        report.knowledge_source_id = source_id
    else:
        out_path = _write_eval_samples(spec, pack, samples)
        report.eval_output_path = str(out_path)

    report.accepted = True
    report.imported_count = len(samples)
    return report


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

@dataclass
class _GateDecision:
    allowed: bool
    reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


def _gate(spec: HFDatasetSpec, metadata: HFDatasetMetadata) -> _GateDecision:
    """Decide whether an import may proceed, with any non-fatal warnings."""
    warnings: List[str] = []

    if spec.dataset_card_required and not metadata.card_present:
        return _GateDecision(
            False, "dataset card required but none was found", warnings)

    # High-risk domains may not ride in under an unknown authority.
    if spec.domain_enum in _HIGH_RISK_DOMAINS \
            and spec.authority_enum is SourceAuthority.UNKNOWN:
        return _GateDecision(
            False,
            f"{spec.domain} domain requires a known authority "
            f"(authority is 'unknown')",
            warnings)

    known = licence_is_known(metadata.license)
    if not known:
        policy = spec.license_policy
        if policy is LicensePolicy.REQUIRE_LICENSE:
            return _GateDecision(
                False, "licence required but none is declared", warnings)
        if policy is LicensePolicy.REJECT_UNKNOWN:
            return _GateDecision(
                False, "unknown licence rejected by policy", warnings)
        # ALLOW_UNKNOWN_FOR_EVAL_ONLY
        if spec.mode is HFDatasetMode.KNOWLEDGE:
            return _GateDecision(
                False,
                "unknown licence cannot enter the knowledge library "
                "(allowed for eval only)",
                warnings)
        warnings.append(
            "unknown licence accepted for EVAL only — verify usage rights "
            "before relying on this data")

    return _GateDecision(True, None, warnings)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _write_knowledge(spec: HFDatasetSpec, pack: ProjectPack,
                     samples: List[HFDatasetSample],
                     metadata: HFDatasetMetadata) -> Optional[str]:
    """Write sampled rows through the frozen KnowledgeLibrary import path."""
    if not samples:
        return None
    safe = _safe_name(spec.dataset_id)
    import_dir = pack.root_path / "hf_imports"
    import_dir.mkdir(parents=True, exist_ok=True)
    doc_path = import_dir / f"{safe}.md"
    doc_path.write_text(_as_markdown(spec, samples), encoding="utf-8")

    library = KnowledgeLibrary(pack.knowledge_library_path)
    source = library.import_text_file(
        doc_path,
        domain=spec.domain_enum,
        authority=spec.authority_enum,
        source_name=f"HF: {spec.dataset_id}",
        version=metadata.revision or spec.revision,
        licence=metadata.license,
        staleness_policy="review_required",
    )
    return source.source_id


def _write_eval_samples(spec: HFDatasetSpec, pack: ProjectPack,
                        samples: List[HFDatasetSample]) -> Path:
    """Write sampled rows to the pack's HF eval samples file (not knowledge)."""
    safe = _safe_name(spec.dataset_id)
    eval_dir = pack.root_path / "hf_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_path = eval_dir / f"{safe}.samples.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
    return out_path


# ---------------------------------------------------------------------------
# Row loading (offline-first)
# ---------------------------------------------------------------------------

def _metadata_for(spec: HFDatasetSpec) -> HFDatasetMetadata:
    return fetch_metadata(
        spec.dataset_id,
        card_path=spec.card_path,
        revision=spec.revision,
        allow_network=spec.allow_network,
    )


def _load_rows(spec: HFDatasetSpec, cap: int) -> List[dict]:
    if spec.local_fixture:
        return _read_jsonl(spec.local_fixture, cap)
    if spec.allow_network:
        return _stream_from_hub(spec, cap)
    raise RuntimeError(
        "no local fixture and network access not enabled: refusing to "
        "download. Set 'local_fixture' or explicitly enable 'allow_network'.")


def _read_jsonl(path: str | Path, cap: int) -> List[dict]:
    rows: List[dict] = []
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if len(rows) >= cap:
                break
            rows.append(json.loads(line))
    return rows


def _stream_from_hub(spec: HFDatasetSpec,
                     cap: int) -> List[dict]:  # pragma: no cover - network path
    """Read a capped, streamed sample from the Hub (opt-in only, never in tests)."""
    from datasets import load_dataset  # type: ignore

    dataset = load_dataset(
        spec.dataset_id,
        spec.config_name,
        split=spec.split,
        streaming=spec.streaming,
        revision=spec.revision,
    )
    rows: List[dict] = []
    for row in dataset:
        if len(rows) >= cap:
            break
        rows.append(dict(row))
    return rows


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _extract_text(row: dict, text_fields: List[str]) -> str:
    parts = [str(row[field_name]).strip()
             for field_name in text_fields
             if row.get(field_name) not in (None, "")]
    return "\n\n".join(p for p in parts if p)


def _as_markdown(spec: HFDatasetSpec, samples: List[HFDatasetSample]) -> str:
    lines = [f"# HF dataset sample: {spec.dataset_id}", ""]
    lines.append(f"Split: {spec.split}; sampled rows: {len(samples)}.")
    lines.append("")
    for sample in samples:
        lines.append(f"## Row {sample.index}")
        lines.append("")
        lines.append(sample.text)
        lines.append("")
    return "\n".join(lines)


def _safe_name(dataset_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_"
                   for c in dataset_id) or "dataset"


def _opt_str(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _str_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(v).strip() for v in value if str(v).strip()]
