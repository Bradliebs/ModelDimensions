"""Governed Hugging Face eval-pack importer — Phase D (v6.9).

This is the first *write* phase of the governed lifecycle, and it writes **eval
material only**. It turns normalized, content-inspected rows (Phase C) into a
deterministic on-disk *eval pack* — a set of retrieval/answer probes — and
nothing else.

Governance honoured here, by construction:

* **Eval is not knowledge.** This importer requires an *eval-intent* request
  validated against an approval that permits eval. It can never write a
  knowledge pack; the knowledge path is a separate, separately-approved phase.
* **Approved is not imported.** A valid approval alone writes nothing. The
  caller must pass an explicit ``pack_dir`` and ``write=True``; the default is a
  dry run that builds the pack in memory and writes nothing.
* **Imported is not activated.** Writing an eval pack does not touch retrieval,
  ranking, grounding, or chat routing. The pack is an inert artefact on disk.
* **Unsafe content is excluded even from eval.** Rows whose Phase C inspection
  blocks eval (unsafe-content findings) are dropped. Possible-PII rows are
  *allowed* in an eval pack — PII caps a row at eval, it does not block eval —
  which is exactly the eval/knowledge asymmetry.
* **Fail closed.** Invalid approval, wrong intent, an existing target directory,
  or no eligible rows all produce a no-write result.

This module imports no writer beyond its own atomic pack write: it cannot touch
the ``MemoryLedger``, the source registry, a proposal, a retrieval index, a
knowledge pack, or an LLM. The docstring is explicit so the import-purity test
can strip it before scanning the body.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from agent.hf_content_inspector import HFContentInspection
from agent.hf_import_lifecycle import (
    HFDatasetApproval,
    HFImportIntent,
    HFImportRequest,
    HFImportValidation,
    validate_import_request,
)
from agent.hf_row_normalizer import HFNormalizationProfile, HFNormalizedRow

EVAL_IMPORTER_VERSION = "hf-eval-pack-v6.9"
DEFAULT_PACK_VERSION = "1.0"
QUESTION_ID_PREFIX = "hfq-"

_PACK_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(now: Optional[datetime]) -> str:
    return (now or _utc_now()).isoformat()


def _sha256_hex(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


# --------------------------------------------------------------------------- #
# Per-profile query / expected role mapping
# --------------------------------------------------------------------------- #

# Each profile declares which canonical role becomes the probe ``query``, which
# becomes the ``expected_answer``, and which remaining roles are carried as
# ``expected_fields`` context. The generic profile has no semantic query role and
# is therefore not eligible for eval-pack import (fail-closed).
_QUERY_ROLE: Dict[HFNormalizationProfile, Tuple[str, ...]] = {
    HFNormalizationProfile.QUESTION_ANSWER: ("question",),
    HFNormalizationProfile.INSTRUCTION_RESPONSE: ("instruction", "input"),
    HFNormalizationProfile.DOCUMENT_TEXT: ("title", "text"),
    HFNormalizationProfile.CLASSIFICATION: ("text",),
    HFNormalizationProfile.RETRIEVAL_PAIR: ("query",),
    HFNormalizationProfile.BENCHMARK_CASE: ("question",),
}
_EXPECTED_ROLE: Dict[HFNormalizationProfile, str] = {
    HFNormalizationProfile.QUESTION_ANSWER: "answer",
    HFNormalizationProfile.INSTRUCTION_RESPONSE: "response",
    HFNormalizationProfile.DOCUMENT_TEXT: "text",
    HFNormalizationProfile.CLASSIFICATION: "label",
    HFNormalizationProfile.RETRIEVAL_PAIR: "passage",
    HFNormalizationProfile.BENCHMARK_CASE: "expected",
}


def profile_supports_eval(profile: HFNormalizationProfile) -> bool:
    return profile in _QUERY_ROLE


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HFEvalPackQuestion:
    """One deterministic eval probe derived from a normalized row."""

    question_id: str
    query: str
    expected_answer: str
    expected_fields: Dict[str, str]
    profile: HFNormalizationProfile
    dataset_id: str
    dataset_revision: str
    split: str
    source_row_id: str
    source_row_key: str
    content_hash: str

    def to_dict(self) -> dict:
        return {
            "_record": "hf_eval_pack_question",
            "question_id": self.question_id,
            "query": self.query,
            "expected_answer": self.expected_answer,
            "expected_fields": dict(self.expected_fields),
            "profile": self.profile.value,
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "split": self.split,
            "source_row_id": self.source_row_id,
            "source_row_key": self.source_row_key,
            "content_hash": self.content_hash,
        }

    def to_retrieval_case(self) -> dict:
        """A retrieval-eval-cases-compatible probe (consumed in Phase F)."""
        return {
            "case_id": self.question_id,
            "query": self.query,
            "expected_sources": [self.dataset_id],
            "minimum_hit_k": 0,
            "tags": ["hf_import", self.profile.value],
            "notes": f"Imported eval probe from {self.dataset_id}@"
                     f"{self.dataset_revision} ({self.split}).",
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HFEvalPackQuestion":
        """Rebuild a question from its serialized ``hf_eval_pack_question`` form."""
        return cls(
            question_id=str(data["question_id"]),
            query=str(data["query"]),
            expected_answer=str(data.get("expected_answer", "")),
            expected_fields=dict(data.get("expected_fields") or {}),
            profile=HFNormalizationProfile(data["profile"]),
            dataset_id=str(data.get("dataset_id", "")),
            dataset_revision=str(data.get("dataset_revision", "")),
            split=str(data.get("split", "")),
            source_row_id=str(data.get("source_row_id", "")),
            source_row_key=str(data.get("source_row_key", "")),
            content_hash=str(data.get("content_hash", "")),
        )


def compute_question_id(pack_id: str, source_row_id: str) -> str:
    """``hfq-<sha256(pack_id|row_id)[:16]>`` — stable per (pack, row)."""
    return QUESTION_ID_PREFIX + _sha256_hex(f"{pack_id}|{source_row_id}")[:16]


@dataclass(frozen=True)
class HFEvalPackManifest:
    """The manifest for an imported HF eval pack (pack-format compatible)."""

    pack_id: str
    pack_version: str
    name: str
    description: str
    dataset_id: str
    dataset_revision: str
    split: str
    profile: HFNormalizationProfile
    approval_id: str
    metadata_fingerprint: str
    intake_assessment_fingerprint: str
    question_count: int
    excluded_unsafe_count: int
    excluded_ineligible_count: int
    importer_version: str
    created_at: str
    pack_hash: str

    def to_dict(self) -> dict:
        return {
            "_record": "hf_eval_pack_manifest",
            # Pack-compatible keys first.
            "pack_id": self.pack_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "pack_kind": "eval",
            # Lineage / provenance.
            "pack_version": self.pack_version,
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "split": self.split,
            "profile": self.profile.value,
            "approval_id": self.approval_id,
            "metadata_fingerprint": self.metadata_fingerprint,
            "intake_assessment_fingerprint": self.intake_assessment_fingerprint,
            "question_count": self.question_count,
            "excluded_unsafe_count": self.excluded_unsafe_count,
            "excluded_ineligible_count": self.excluded_ineligible_count,
            "importer_version": self.importer_version,
            "pack_hash": self.pack_hash,
        }


class HFEvalImportStatus(str, Enum):
    """The outcome of an eval-pack import (dry-run or write)."""

    IMPORT_READY = "import_ready"  # valid dry run; nothing written
    IMPORT_WRITTEN = "import_written"  # pack written to disk
    IMPORT_BLOCKED = "import_blocked"  # approval/intent invalid
    NO_ELIGIBLE_ROWS = "no_eligible_rows"  # nothing left after filtering
    PACK_EXISTS = "pack_exists"  # target dir already exists
    INVALID_PACK_ID = "invalid_pack_id"


@dataclass(frozen=True)
class HFEvalPackImportResult:
    """The full outcome of an eval-pack import."""

    status: HFEvalImportStatus
    written: bool
    pack_dir: Optional[str]
    written_paths: Tuple[str, ...]
    questions: Tuple[HFEvalPackQuestion, ...]
    manifest: Optional[HFEvalPackManifest]
    validation: Optional[HFImportValidation]
    message: str
    excluded_unsafe_row_ids: Tuple[str, ...] = ()
    excluded_ineligible_row_ids: Tuple[str, ...] = ()

    @property
    def question_count(self) -> int:
        return len(self.questions)

    @property
    def ok(self) -> bool:
        return self.status in (HFEvalImportStatus.IMPORT_READY,
                               HFEvalImportStatus.IMPORT_WRITTEN)

    def to_dict(self) -> dict:
        return {
            "_record": "hf_eval_pack_import_result",
            "status": self.status.value,
            "written": self.written,
            "pack_dir": self.pack_dir,
            "written_paths": list(self.written_paths),
            "question_count": self.question_count,
            "message": self.message,
            "excluded_unsafe_row_ids": list(self.excluded_unsafe_row_ids),
            "excluded_ineligible_row_ids": list(self.excluded_ineligible_row_ids),
            "validation": self.validation.to_dict() if self.validation else None,
            "manifest": self.manifest.to_dict() if self.manifest else None,
            "questions": [q.to_dict() for q in self.questions],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


@dataclass(frozen=True)
class HFEvalPackImportRequest:
    """An explicit request to build (and optionally write) an HF eval pack."""

    pack_id: str
    approval: HFDatasetApproval
    request: HFImportRequest
    normalized_rows: Tuple[HFNormalizedRow, ...]
    inspections: Tuple[HFContentInspection, ...] = ()
    pack_dir: Optional[str] = None
    write: bool = False
    pack_name: str = ""
    pack_description: str = ""
    metadata: object = None
    assessment: object = None
    column_query_override: Optional[str] = None


# --------------------------------------------------------------------------- #
# Building questions (pure; no writes)
# --------------------------------------------------------------------------- #


def _first_present(row: HFNormalizedRow, roles: Sequence[str]) -> str:
    for role in roles:
        text = row.normalized_fields.get(role, "")
        if text:
            return text
    return ""


def build_question(
    row: HFNormalizedRow, *, pack_id: str,
) -> Optional[HFEvalPackQuestion]:
    """Derive one eval probe from a normalized row, or ``None`` if ineligible."""
    if not profile_supports_eval(row.profile):
        return None
    query_roles = _QUERY_ROLE[row.profile]
    expected_role = _EXPECTED_ROLE[row.profile]
    query = _first_present(row, query_roles)
    expected = row.normalized_fields.get(expected_role, "")
    if not query or not expected:
        return None
    used_query_role = next((r for r in query_roles
                            if row.normalized_fields.get(r)), "")
    extra = {role: text for role, text in row.normalized_fields.items()
             if role != used_query_role and role != expected_role}
    return HFEvalPackQuestion(
        question_id=compute_question_id(pack_id, row.row_id),
        query=query,
        expected_answer=expected,
        expected_fields=dict(sorted(extra.items())),
        profile=row.profile,
        dataset_id=row.dataset_id,
        dataset_revision=row.dataset_revision,
        split=row.split,
        source_row_id=row.row_id,
        source_row_key=row.source_row_key,
        content_hash=row.content_hash,
    )


def _pack_hash(
    pack_id: str,
    approval: HFDatasetApproval,
    request: HFImportRequest,
    questions: Sequence[HFEvalPackQuestion],
) -> str:
    payload = {
        "pack_id": pack_id,
        "dataset_id": approval.dataset_id,
        "dataset_revision": approval.dataset_revision,
        "split": request.split,
        "approval_id": approval.approval_id,
        "metadata_fingerprint": approval.metadata_fingerprint,
        "intake_assessment_fingerprint": approval.intake_assessment_fingerprint,
        "importer_version": EVAL_IMPORTER_VERSION,
        "questions": [q.to_dict() for q in questions],
    }
    return "hfevalpack-" + _sha256_hex(_canonical_json(payload))[:16]


# --------------------------------------------------------------------------- #
# Atomic pack write
# --------------------------------------------------------------------------- #


def _atomic_write_pack(pack_dir: Path, files: Mapping[str, str]) -> List[str]:
    """Write ``files`` into ``pack_dir`` atomically; never leave a partial pack."""
    parent = pack_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(
        prefix=f".{pack_dir.name}.hfeval-", dir=str(parent)))
    try:
        for name, content in files.items():
            (tmp_dir / name).write_text(content, encoding="utf-8")
        os.rename(tmp_dir, pack_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return [str(pack_dir / name) for name in files]


def _render_pack_files(
    manifest: HFEvalPackManifest,
    questions: Sequence[HFEvalPackQuestion],
) -> Dict[str, str]:
    manifest_json = json.dumps(
        manifest.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)
    questions_jsonl = "\n".join(
        json.dumps(q.to_dict(), ensure_ascii=False) for q in questions)
    if questions_jsonl:
        questions_jsonl += "\n"
    return {
        "manifest.json": manifest_json + "\n",
        "eval_questions.jsonl": questions_jsonl,
    }


# --------------------------------------------------------------------------- #
# Top-level import entry point
# --------------------------------------------------------------------------- #


def import_eval_pack(
    request: HFEvalPackImportRequest, *, now: Optional[datetime] = None,
) -> HFEvalPackImportResult:
    """Validate an eval-intent request and, only when asked, write the eval pack.

    Fail-closed order: pack-id shape, then approval/intent validation, then
    eligible-row filtering (unsafe rows excluded, ineligible profiles/rows
    dropped). Dry run (``write=False``) builds everything and writes nothing;
    ``write=True`` with a non-existent ``pack_dir`` writes atomically.
    """
    created_at = _iso(now)

    if not _PACK_ID_RE.match(request.pack_id or ""):
        return HFEvalPackImportResult(
            status=HFEvalImportStatus.INVALID_PACK_ID, written=False,
            pack_dir=None, written_paths=(), questions=(), manifest=None,
            validation=None,
            message=f"invalid pack id {request.pack_id!r}")

    # Eval pack requires an eval-intent request.
    if request.request.intent is not HFImportIntent.EVAL:
        return HFEvalPackImportResult(
            status=HFEvalImportStatus.IMPORT_BLOCKED, written=False,
            pack_dir=None, written_paths=(), questions=(), manifest=None,
            validation=None,
            message="eval-pack import requires an eval-intent request")

    validation = validate_import_request(
        request.approval, request.request,
        metadata=request.metadata, assessment=request.assessment, now=now)
    if not validation.valid:
        return HFEvalPackImportResult(
            status=HFEvalImportStatus.IMPORT_BLOCKED, written=False,
            pack_dir=None, written_paths=(), questions=(), manifest=None,
            validation=validation,
            message="; ".join(validation.messages))

    # Filter rows: drop those whose content inspection blocks eval (unsafe).
    blocks_eval: Dict[str, bool] = {
        i.row_id: i.blocks_eval for i in request.inspections}
    excluded_unsafe: List[str] = []
    excluded_ineligible: List[str] = []
    questions: List[HFEvalPackQuestion] = []
    for row in request.normalized_rows:
        if blocks_eval.get(row.row_id, False):
            excluded_unsafe.append(row.row_id)
            continue
        question = build_question(row, pack_id=request.pack_id)
        if question is None:
            excluded_ineligible.append(row.row_id)
            continue
        questions.append(question)

    if not questions:
        return HFEvalPackImportResult(
            status=HFEvalImportStatus.NO_ELIGIBLE_ROWS, written=False,
            pack_dir=None, written_paths=(), questions=(), manifest=None,
            validation=validation,
            message="no eligible rows remained after filtering",
            excluded_unsafe_row_ids=tuple(excluded_unsafe),
            excluded_ineligible_row_ids=tuple(excluded_ineligible))

    profile = questions[0].profile
    manifest = HFEvalPackManifest(
        pack_id=request.pack_id,
        pack_version=DEFAULT_PACK_VERSION,
        name=request.pack_name or f"HF eval pack: {request.approval.dataset_id}",
        description=request.pack_description
        or f"Governed eval pack imported from {request.approval.dataset_id}@"
           f"{request.approval.dataset_revision}.",
        dataset_id=request.approval.dataset_id,
        dataset_revision=request.approval.dataset_revision,
        split=request.request.split,
        profile=profile,
        approval_id=request.approval.approval_id,
        metadata_fingerprint=request.approval.metadata_fingerprint,
        intake_assessment_fingerprint=request.approval.intake_assessment_fingerprint,
        question_count=len(questions),
        excluded_unsafe_count=len(excluded_unsafe),
        excluded_ineligible_count=len(excluded_ineligible),
        importer_version=EVAL_IMPORTER_VERSION,
        created_at=created_at,
        pack_hash=_pack_hash(request.pack_id, request.approval, request.request,
                             questions),
    )

    if not request.write:
        return HFEvalPackImportResult(
            status=HFEvalImportStatus.IMPORT_READY, written=False,
            pack_dir=request.pack_dir, written_paths=(),
            questions=tuple(questions), manifest=manifest,
            validation=validation,
            message=f"dry run: {len(questions)} eval question(s) ready; "
                    "nothing written",
            excluded_unsafe_row_ids=tuple(excluded_unsafe),
            excluded_ineligible_row_ids=tuple(excluded_ineligible))

    if not request.pack_dir:
        return HFEvalPackImportResult(
            status=HFEvalImportStatus.IMPORT_BLOCKED, written=False,
            pack_dir=None, written_paths=(), questions=tuple(questions),
            manifest=manifest, validation=validation,
            message="write requested but no pack_dir given",
            excluded_unsafe_row_ids=tuple(excluded_unsafe),
            excluded_ineligible_row_ids=tuple(excluded_ineligible))

    pack_dir = Path(request.pack_dir)
    if pack_dir.exists():
        return HFEvalPackImportResult(
            status=HFEvalImportStatus.PACK_EXISTS, written=False,
            pack_dir=str(pack_dir), written_paths=(),
            questions=tuple(questions), manifest=manifest,
            validation=validation,
            message=f"target pack dir already exists: {pack_dir}",
            excluded_unsafe_row_ids=tuple(excluded_unsafe),
            excluded_ineligible_row_ids=tuple(excluded_ineligible))

    files = _render_pack_files(manifest, questions)
    written_paths = _atomic_write_pack(pack_dir, files)
    return HFEvalPackImportResult(
        status=HFEvalImportStatus.IMPORT_WRITTEN, written=True,
        pack_dir=str(pack_dir), written_paths=tuple(written_paths),
        questions=tuple(questions), manifest=manifest, validation=validation,
        message=f"wrote eval pack with {len(questions)} question(s)",
        excluded_unsafe_row_ids=tuple(excluded_unsafe),
        excluded_ineligible_row_ids=tuple(excluded_ineligible))


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def render_eval_import_markdown(result: HFEvalPackImportResult) -> str:
    """Deterministic summary. Never prints raw query/answer content."""
    manifest = result.manifest
    lines = [
        "# Governed Hugging Face eval-pack import (v6.9)",
        "",
        f"- status: **{result.status.value}**",
        f"- written: {result.written}",
        f"- questions: {result.question_count}",
        f"- excluded (unsafe): {len(result.excluded_unsafe_row_ids)}",
        f"- excluded (ineligible): {len(result.excluded_ineligible_row_ids)}",
    ]
    if manifest is not None:
        lines += [
            f"- pack id: `{manifest.pack_id}`",
            f"- dataset: `{manifest.dataset_id}@{manifest.dataset_revision}`",
            f"- split: `{manifest.split}`",
            f"- profile: `{manifest.profile.value}`",
            f"- pack hash: `{manifest.pack_hash}`",
        ]
    if result.message:
        lines += ["", f"> {result.message}"]
    return "\n".join(lines) + "\n"
