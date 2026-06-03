"""Curated knowledge pack builder (v1.8).

A *curated knowledge pack* is a project pack whose knowledge library is built
deliberately from named, provenance-tagged sources — not from a random dump of
scraped text. This module turns a small, declarative *build plan* into imported
knowledge inside a pack, recording exactly what was imported (and what was
deliberately skipped) so the result is auditable before anyone relies on it.

Design constraints (v1.8):

* **Local files first.** A source points at a local file or a directory of
  files. URL ingestion is *scaffolded but disabled*: a URL source is recorded as
  skipped unless ingestion is explicitly enabled, so the network is never
  touched silently.
* **No new geometry.** Importing reuses the frozen
  :meth:`KnowledgeLibrary.import_text_file` path; this module only decides which
  files become which sources, with which provenance.
* **Auditable.** Building returns a :class:`PackBuildReport` listing every source
  (name, domain, authority, version, licence, chunk count) and every skip.

The dataclasses for *evaluating* a built pack (:class:`PackEvalQuestion`,
:class:`PackEvalResult`) live here too so the report can carry retrieval
validation results; the evaluation logic itself is in
:mod:`agent.pack_evaluator`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from agent.knowledge_sources import KnowledgeDomain, SourceAuthority
from agent.project_packs import PackRegistry, ProjectPack
from agent.workbench_service import WorkbenchService

# File patterns a directory source imports by default (text/Markdown only).
_DEFAULT_INCLUDE = ("*.md", "*.markdown", "*.txt")

# Recorded as the skip reason whenever a URL source is left out because URL
# ingestion was not explicitly enabled. Surfacing this keeps a disabled fetch
# from being mistaken for "nothing matched".
URL_INGESTION_DISABLED_REASON = (
    "URL ingestion is disabled (local sources only); enable it explicitly to "
    "fetch remote sources"
)

# Recorded as the skip reason whenever a Hugging Face dataset source is left out
# because HF ingestion was not explicitly enabled. Like URL ingestion, HF
# sampling never happens during a normal (starter) build.
HF_INGESTION_DISABLED_REASON = (
    "Hugging Face ingestion is disabled; import HF datasets explicitly with the "
    "hf importer (free does not mean trusted)"
)

# The ``source_type`` value that marks a source as a Hugging Face dataset sample
# rather than a local file/directory.
HF_SOURCE_TYPE = "huggingface_dataset"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_url(path_or_url: str) -> bool:
    """True for ``http://`` / ``https://`` sources (everything else is local)."""
    lowered = (path_or_url or "").strip().lower()
    return lowered.startswith("http://") or lowered.startswith("https://")


# ---------------------------------------------------------------------------
# Source / plan specs
# ---------------------------------------------------------------------------

@dataclass
class PackSourceSpec:
    """One source to import into a pack, with its full provenance.

    ``path_or_url`` may be a local file, a local directory (imported using
    ``include_patterns`` / ``exclude_patterns``), or an ``http(s)`` URL (only
    used when URL ingestion is explicitly enabled).
    """

    source_name: str
    path_or_url: str
    domain: str
    authority: str
    version: Optional[str] = None
    licence: Optional[str] = None
    retrieved_at: str = field(default_factory=_utc_now)
    staleness_policy: str = "static"
    include_patterns: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None
    # Optional Hugging Face dataset source (v1.9). When ``source_type`` is
    # ``"huggingface_dataset"`` the source is sampled from the Hub instead of a
    # local file, and only when HF ingestion is explicitly enabled.
    source_type: str = "local"
    dataset_id: Optional[str] = None
    split: Optional[str] = None
    text_fields: Optional[List[str]] = None
    sample_size: Optional[int] = None
    mode: Optional[str] = None

    @property
    def is_url(self) -> bool:
        return _is_url(self.path_or_url)

    @property
    def is_hf(self) -> bool:
        return (self.source_type or "local").lower() == HF_SOURCE_TYPE

    def to_dict(self) -> dict:
        return {
            "source_name": self.source_name,
            "path_or_url": self.path_or_url,
            "domain": self.domain,
            "authority": self.authority,
            "version": self.version,
            "licence": self.licence,
            "retrieved_at": self.retrieved_at,
            "staleness_policy": self.staleness_policy,
            "include_patterns": list(self.include_patterns)
            if self.include_patterns else None,
            "exclude_patterns": list(self.exclude_patterns)
            if self.exclude_patterns else None,
            "source_type": self.source_type,
            "dataset_id": self.dataset_id,
            "split": self.split,
            "text_fields": list(self.text_fields) if self.text_fields else None,
            "sample_size": self.sample_size,
            "mode": self.mode,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PackSourceSpec":
        if "source_name" not in data:
            raise ValueError("source spec needs a 'source_name'")
        source_type = str(data.get("source_type", "local"))
        is_hf = source_type.lower() == HF_SOURCE_TYPE
        if "path_or_url" not in data and not is_hf:
            raise ValueError(
                f"source {data['source_name']!r} needs a 'path_or_url'")
        if is_hf and "dataset_id" not in data:
            raise ValueError(
                f"source {data['source_name']!r} needs a 'dataset_id' for a "
                f"{HF_SOURCE_TYPE} source")
        if "domain" not in data:
            raise ValueError(
                f"source {data['source_name']!r} needs a 'domain'")
        if "authority" not in data:
            raise ValueError(
                f"source {data['source_name']!r} needs an 'authority'")
        path_or_url = str(
            data.get("path_or_url")
            or (f"hf://{data['dataset_id']}" if is_hf else ""))
        return cls(
            source_name=str(data["source_name"]),
            path_or_url=path_or_url,
            domain=str(data["domain"]),
            authority=str(data["authority"]),
            version=_opt_str(data.get("version")),
            licence=_opt_str(data.get("licence")),
            retrieved_at=str(data.get("retrieved_at") or _utc_now()),
            staleness_policy=str(data.get("staleness_policy", "static")),
            include_patterns=_opt_str_list(data.get("include_patterns")),
            exclude_patterns=_opt_str_list(data.get("exclude_patterns")),
            source_type=source_type,
            dataset_id=_opt_str(data.get("dataset_id")),
            split=_opt_str(data.get("split")),
            text_fields=_opt_str_list(data.get("text_fields")),
            sample_size=(int(data["sample_size"])
                         if data.get("sample_size") is not None else None),
            mode=_opt_str(data.get("mode")),
        )


@dataclass
class PackBuildPlan:
    """A declarative recipe: which pack, which sources, with which defaults."""

    pack_name: str
    description: str = ""
    default_domain: Optional[str] = None
    default_knowledge_backend: str = "deterministic"
    allow_url_ingestion: bool = False
    enabled: bool = True
    sources: List[PackSourceSpec] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pack_name": self.pack_name,
            "description": self.description,
            "default_domain": self.default_domain,
            "default_knowledge_backend": self.default_knowledge_backend,
            "allow_url_ingestion": self.allow_url_ingestion,
            "enabled": self.enabled,
            "sources": [s.to_dict() for s in self.sources],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PackBuildPlan":
        if "pack_name" not in data:
            raise ValueError("build plan needs a 'pack_name'")
        sources = [PackSourceSpec.from_dict(s) for s in data.get("sources", [])]
        return cls(
            pack_name=str(data["pack_name"]),
            description=str(data.get("description", "")),
            default_domain=_opt_str(data.get("default_domain")),
            default_knowledge_backend=str(
                data.get("default_knowledge_backend", "deterministic")),
            allow_url_ingestion=bool(data.get("allow_url_ingestion", False)),
            enabled=bool(data.get("enabled", True)),
            sources=sources,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "PackBuildPlan":
        """Load a plan from a JSON or YAML spec file.

        YAML requires PyYAML; a ``.json`` spec needs no extra dependency. JSON is
        valid YAML, so a ``.json`` spec also loads under the YAML path.
        """
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            data = _load_yaml(text, path)
        else:
            data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError(f"spec {path} must be a mapping at the top level")
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# Evaluation specs (logic lives in agent.pack_evaluator)
# ---------------------------------------------------------------------------

@dataclass
class PackEvalQuestion:
    """One retrieval/source check to run against a built pack."""

    query: str
    question_id: Optional[str] = None
    expected_domain: Optional[str] = None
    expected_source: Optional[str] = None
    expected_authority: Optional[str] = None
    expected_chunk_contains: Optional[str] = None
    forbidden_terms: List[str] = field(default_factory=list)
    required_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "question_id": self.question_id,
            "expected_domain": self.expected_domain,
            "expected_source": self.expected_source,
            "expected_authority": self.expected_authority,
            "expected_chunk_contains": self.expected_chunk_contains,
            "forbidden_terms": list(self.forbidden_terms),
            "required_flags": list(self.required_flags),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PackEvalQuestion":
        if "query" not in data:
            raise ValueError("eval question needs a 'query'")
        return cls(
            query=str(data["query"]),
            question_id=_opt_str(data.get("question_id") or data.get("id")),
            expected_domain=_opt_str(data.get("expected_domain")),
            expected_source=_opt_str(data.get("expected_source")),
            expected_authority=_opt_str(data.get("expected_authority")),
            expected_chunk_contains=_opt_str(
                data.get("expected_chunk_contains")),
            forbidden_terms=_opt_str_list(data.get("forbidden_terms")) or [],
            required_flags=_opt_str_list(data.get("required_flags")) or [],
        )


@dataclass
class PackEvalResult:
    """The outcome of one :class:`PackEvalQuestion` against a pack."""

    query: str
    passed: bool
    reason: str
    question_id: Optional[str] = None
    checks: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "question_id": self.question_id,
            "passed": self.passed,
            "reason": self.reason,
            "checks": list(self.checks),
        }


# ---------------------------------------------------------------------------
# Build report
# ---------------------------------------------------------------------------

@dataclass
class PackBuildReport:
    """What a build produced: imported sources, skips, and (optional) eval."""

    pack_id: str
    pack_name: str
    sources: List[dict] = field(default_factory=list)
    skipped: List[dict] = field(default_factory=list)
    eval_results: List[PackEvalResult] = field(default_factory=list)

    @property
    def source_count(self) -> int:
        return len(self.sources)

    @property
    def total_chunks(self) -> int:
        return sum(int(s.get("chunks", 0)) for s in self.sources)

    def eval_summary(self) -> Dict[str, int]:
        passed = sum(1 for r in self.eval_results if r.passed)
        return {"passed": passed, "total": len(self.eval_results)}

    def with_eval(self, results: List[PackEvalResult]) -> "PackBuildReport":
        self.eval_results = list(results)
        return self

    def to_dict(self) -> dict:
        return {
            "pack_id": self.pack_id,
            "pack_name": self.pack_name,
            "source_count": self.source_count,
            "total_chunks": self.total_chunks,
            "sources": list(self.sources),
            "skipped": list(self.skipped),
            "eval": {
                **self.eval_summary(),
                "results": [r.to_dict() for r in self.eval_results],
            },
        }


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def open_pack_service(registry: PackRegistry,
                      pack_id_or_name: str) -> WorkbenchService:
    """Open a service bound to an existing pack (loads its knowledge from disk)."""
    pack = registry.get_pack(pack_id_or_name)
    if pack is None:
        raise KeyError(f"no such pack: {pack_id_or_name!r}")
    return WorkbenchService.from_pack(pack, registry=registry)


def _ensure_pack(plan: PackBuildPlan, registry: PackRegistry) -> ProjectPack:
    pack = registry.get_pack(plan.pack_name)
    if pack is not None:
        return pack
    return registry.create_pack(
        plan.pack_name,
        description=plan.description,
        default_knowledge_backend=plan.default_knowledge_backend,
        default_domain=plan.default_domain,
    )


def _resolve_local_files(spec: PackSourceSpec) -> List[Path]:
    """Resolve a local source spec to the concrete files it imports.

    A file path yields itself; a directory yields its matching files (using
    ``include_patterns`` / ``exclude_patterns``). Raises ``FileNotFoundError``
    if the path does not exist so a typo in a spec fails loudly.
    """
    base = Path(spec.path_or_url)
    if not base.exists():
        raise FileNotFoundError(spec.path_or_url)
    if base.is_file():
        return [base]
    includes = spec.include_patterns or list(_DEFAULT_INCLUDE)
    excludes = spec.exclude_patterns or []
    matched: Dict[str, Path] = {}
    for pattern in includes:
        for found in base.rglob(pattern):
            if found.is_file():
                matched[str(found)] = found
    for pattern in excludes:
        for found in base.rglob(pattern):
            matched.pop(str(found), None)
    return [matched[key] for key in sorted(matched)]


def _ingest_url(spec: PackSourceSpec) -> None:  # pragma: no cover - scaffold
    """Placeholder for URL ingestion (intentionally not implemented in v1.8)."""
    raise NotImplementedError(
        "URL ingestion is scaffolded but not implemented; provide local files "
        "for this source instead")


def build_pack(plan: PackBuildPlan, registry: PackRegistry, *,
               allow_url_ingestion: bool = False,
               allow_hf_ingestion: bool = False) -> PackBuildReport:
    """Build (or extend) a pack's knowledge library from a plan's sources.

    Only local sources are imported by default. A URL source is recorded as
    skipped unless URL ingestion is enabled, and a Hugging Face dataset source
    is recorded as skipped unless HF ingestion is *explicitly* enabled — so
    neither a network fetch nor a Hub sample can happen during a normal build.
    Re-importing the same file is idempotent (the source id is derived from
    name + path). The active pack selection is not changed.
    """
    pack = _ensure_pack(plan, registry)
    service = WorkbenchService.from_pack(pack, registry=registry)
    report = PackBuildReport(pack_id=pack.pack_id, pack_name=pack.name)

    url_enabled = allow_url_ingestion and plan.allow_url_ingestion

    for spec in plan.sources:
        if spec.is_hf:
            _build_hf_source(spec, pack, service, report,
                             allow_hf_ingestion=allow_hf_ingestion)
            continue
        if spec.is_url and not url_enabled:
            report.skipped.append({
                "source_name": spec.source_name,
                "path_or_url": spec.path_or_url,
                "reason": URL_INGESTION_DISABLED_REASON,
            })
            continue
        if spec.is_url:
            _ingest_url(spec)  # explicit opt-in; raises until implemented
            continue
        try:
            files = _resolve_local_files(spec)
        except FileNotFoundError:
            report.skipped.append({
                "source_name": spec.source_name,
                "path_or_url": spec.path_or_url,
                "reason": "local path not found",
            })
            continue
        if not files:
            report.skipped.append({
                "source_name": spec.source_name,
                "path_or_url": spec.path_or_url,
                "reason": "no files matched include/exclude patterns",
            })
            continue
        for file_path in files:
            source = service.knowledge.import_text_file(
                file_path,
                domain=KnowledgeDomain(spec.domain),
                authority=SourceAuthority(spec.authority),
                source_name=spec.source_name,
                version=spec.version,
                licence=spec.licence,
                staleness_policy=spec.staleness_policy,
            )
            chunks = len(service.knowledge.list_chunks(
                source_id=source.source_id))
            report.sources.append({
                "source_id": source.source_id,
                "source_name": source.source_name,
                "domain": source.domain.value,
                "authority": source.authority.value,
                "version": source.version,
                "licence": source.licence,
                "staleness_policy": source.staleness_policy,
                "path_or_url": source.path_or_url,
                "chunks": chunks,
            })
    return report


def _build_hf_source(spec: PackSourceSpec, pack, service: WorkbenchService,
                     report: PackBuildReport, *,
                     allow_hf_ingestion: bool) -> None:
    """Handle a Hugging Face dataset source during a build (gated, opt-in).

    Disabled by default: the source is recorded as skipped so a normal/starter
    build never samples the Hub. When enabled, the dataset is imported through
    the dedicated licence-aware importer and the outcome is recorded.
    """
    if not allow_hf_ingestion:
        report.skipped.append({
            "source_name": spec.source_name,
            "path_or_url": spec.path_or_url,
            "reason": HF_INGESTION_DISABLED_REASON,
        })
        return

    # Local import keeps the importer dependency out of the default build path.
    from agent.hf_dataset_importer import (
        HFDatasetMode,
        HFDatasetSpec,
        import_hf_dataset_to_pack,
    )

    hf_spec = HFDatasetSpec(
        dataset_id=spec.dataset_id or "",
        split=spec.split or "train",
        text_fields=list(spec.text_fields) if spec.text_fields else ["text"],
        sample_size=spec.sample_size if spec.sample_size is not None else 100,
        domain=spec.domain,
        authority=spec.authority,
        mode=HFDatasetMode(spec.mode) if spec.mode else HFDatasetMode.EVAL,
        revision=spec.version,
    )
    hf_report = import_hf_dataset_to_pack(hf_spec, pack)
    if not hf_report.accepted:
        report.skipped.append({
            "source_name": spec.source_name,
            "path_or_url": spec.path_or_url,
            "reason": hf_report.rejected_reason or "HF import refused",
        })
        return

    if hf_report.knowledge_source_id:
        service.knowledge.load()
        chunks = len(service.knowledge.list_chunks(
            source_id=hf_report.knowledge_source_id))
        report.sources.append({
            "source_id": hf_report.knowledge_source_id,
            "source_name": spec.source_name,
            "domain": spec.domain,
            "authority": spec.authority,
            "version": hf_report.license,
            "licence": hf_report.license,
            "staleness_policy": "review_required",
            "path_or_url": spec.path_or_url,
            "chunks": chunks,
        })
    else:
        report.skipped.append({
            "source_name": spec.source_name,
            "path_or_url": spec.path_or_url,
            "reason": f"eval samples written to {hf_report.eval_output_path}",
        })


def report_for_pack(service: WorkbenchService, *,
                    pack_id: Optional[str] = None,
                    pack_name: Optional[str] = None,
                    eval_results: Optional[List[PackEvalResult]] = None
                    ) -> PackBuildReport:
    """Build a report from a pack's *current* knowledge (no rebuild).

    Used by ``pack-report``: it inventories the active sources and, when eval
    results are supplied, includes the pass/fail outcomes.
    """
    info = service.active_pack_info() or {}
    report = PackBuildReport(
        pack_id=pack_id or info.get("pack_id", ""),
        pack_name=pack_name or info.get("name", ""),
    )
    for row in service.list_knowledge_sources():
        report.sources.append({
            "source_id": row["source_id"],
            "source_name": row["source_name"],
            "domain": row["domain"],
            "authority": row["authority"],
            "version": row.get("version"),
            "path_or_url": row.get("path_or_url"),
            "chunks": row.get("chunks", 0),
        })
    if eval_results is not None:
        report.with_eval(eval_results)
    return report


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _opt_str(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _opt_str_list(value) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _load_yaml(text: str, path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            f"{path.suffix} spec needs PyYAML; install it or use a .json spec"
        ) from exc
    return yaml.safe_load(text)
