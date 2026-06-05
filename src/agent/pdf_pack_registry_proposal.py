"""v6.8 Imported PDF Pack -> Source Registry Entry proposal (proposal-only).

An approved PDF import (v6.6) writes a knowledge pack, and v6.7 measures whether
that pack helps retrieval. But an imported pack is still *untracked* by the
source registry (v4.x) — the place that records a source's authority, owner,
freshness, supersession, and review lineage. This module bridges that gap with a
**proposal only**: it reads an imported pack's ``manifest.json`` and proposes a
:class:`~agent.source_registry.SourceRegistryEntry` that a human can review and,
if they choose, paste into the registry.

It is deliberately inert and read-only:

* It **proposes, never applies.** The output is a proposal object whose
  ``requires_human_approval`` is always ``True`` and whose ``status`` is always
  ``"proposed"``. Nothing is written to the registry — ``save_registry`` is the
  registry's only writer and is never imported or called here.
* It **reads the pack manifest only.** It does not re-parse the PDF, re-import
  chunks, touch the knowledge library, run OCR, or call any model.
* It **mutates no durable state.** No memory ledger, no source registry, no
  review queue, no proposal application, and no retrieval/ranking/source-
  selection/grounding/composer/chat-routing behaviour changes.
* **Provenance and permission stay operator-declared.** The manifest's declared
  authority, provenance, and permission are surfaced as *proposed* metadata and
  flagged for review when missing or unrecognised; this layer never upgrades a
  source's trust on its own. A pack with gaps is proposed as a ``draft`` for a
  human to ratify, not an ``active`` source.

The proposal carries a ready-to-paste registry entry (the proposed
``SourceRegistryEntry``), the pack's content lineage (manifest hash, source file
hash, preview fingerprint, approval id), and review findings — so the human has
everything needed to decide, in one deterministic record.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from agent.source_registry import (
    AuthorityLevel,
    FindingSeverity,
    SourceRegistryEntry,
    SourceStatus,
    index_by_id,
    load_registry,
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

PROPOSAL_LAYER_VERSION = "v6.8"
SUPPORTED_MANIFEST_RECORD = "pdf_knowledge_pack_manifest"
MANIFEST_FILENAME = "manifest.json"
PROPOSED_SOURCE_TYPE = "pdf_knowledge_pack"

# Recognised authority strings (mirrors AuthorityLevel); anything else is
# proposed as ``unknown`` and flagged for review.
_KNOWN_AUTHORITIES = {a.value for a in AuthorityLevel}

# Deterministic ordering for findings: errors first, then warnings, then info.
_SEVERITY_RANK: Dict[FindingSeverity, int] = {
    FindingSeverity.ERROR: 0,
    FindingSeverity.WARNING: 1,
    FindingSeverity.INFO: 2,
}


class ProposalFindingCode:
    """Stable string codes for registry-proposal review findings."""

    # lineage / declared metadata gaps
    UNKNOWN_PACK_AUTHORITY = "unknown_pack_authority"
    MISSING_PROVENANCE = "missing_provenance"
    MISSING_PERMISSION = "missing_permission"
    EVAL_ONLY_INTENT = "eval_only_intent"
    UNRESOLVED_IMPORT_FINDINGS = "unresolved_import_findings"
    EXCLUDED_CHUNKS_PRESENT = "excluded_chunks_present"
    # registry integration
    DUPLICATE_SOURCE_ID = "duplicate_source_id"
    COMPLETE_LINEAGE = "complete_lineage"


# What the proposal recommends a human do with the pack.
class ProposalDecision:
    """Top-level recommendation derived from the worst review finding."""

    PROPOSE_NEW_ACTIVE = "propose_new_active"
    PROPOSE_NEW_DRAFT = "propose_new_draft"
    CONFLICT_EXISTING_SOURCE = "conflict_existing_source"


# -----------------------------------------------------------------------------
# Data records (all frozen; pure data)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class RegistryProposalFinding:
    """One review note about the proposed registry entry. Inert data."""

    code: str
    severity: FindingSeverity
    message: str

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "_record": "pdf_pack_registry_proposal_finding",
        }


@dataclass(frozen=True)
class PdfPackRegistryProposal:
    """A proposed source-registry entry derived from an imported PDF pack.

    Pure data and never applied: ``requires_human_approval`` is always ``True``
    and ``status`` is always ``"proposed"``. ``proposed_entry`` is a complete
    :class:`SourceRegistryEntry` a human can paste into the registry after
    review; ``findings`` explain why it is proposed ``active`` or ``draft`` (or
    flagged as a conflict).
    """

    proposal_id: str
    pack_id: str
    decision: str
    proposed_entry: SourceRegistryEntry
    findings: List[RegistryProposalFinding]
    manifest_hash: str
    source_file_hash: str
    preview_fingerprint: str
    approval_id: str
    requires_human_approval: bool = True
    status: str = "proposed"
    created_at: Optional[str] = None

    @property
    def proposed_status(self) -> SourceStatus:
        return self.proposed_entry.status

    @property
    def has_conflict(self) -> bool:
        return self.decision == ProposalDecision.CONFLICT_EXISTING_SOURCE

    def to_dict(self) -> dict:
        return {
            "proposal_id": self.proposal_id,
            "pack_id": self.pack_id,
            "decision": self.decision,
            "proposed_entry": self.proposed_entry.to_dict(),
            "findings": [f.to_dict() for f in self.findings],
            "manifest_hash": self.manifest_hash,
            "source_file_hash": self.source_file_hash,
            "preview_fingerprint": self.preview_fingerprint,
            "approval_id": self.approval_id,
            "requires_human_approval": self.requires_human_approval,
            "status": self.status,
            "created_at": self.created_at,
            "_record": "pdf_pack_registry_proposal",
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


# -----------------------------------------------------------------------------
# Loading (pure reads)
# -----------------------------------------------------------------------------

def load_pack_manifest(path: str | Path) -> dict:
    """Load an imported pack's manifest dict from a pack dir or a manifest file.

    If ``path`` is a directory, its ``manifest.json`` is read; if it is a file,
    that file is read directly. The record type is validated so a non-pack JSON
    fails cleanly. A pure read: nothing else on disk is touched.
    """
    p = Path(path)
    manifest_path = p / MANIFEST_FILENAME if p.is_dir() else p
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(
            f"{manifest_path} is not a single JSON object manifest")
    record = data.get("_record")
    if record != SUPPORTED_MANIFEST_RECORD:
        raise ValueError(
            f"{manifest_path} is not a {SUPPORTED_MANIFEST_RECORD!r} manifest "
            f"(found _record={record!r})")
    return data


# -----------------------------------------------------------------------------
# Proposal generation (pure)
# -----------------------------------------------------------------------------

def _proposal_id(pack_id: str, manifest_hash: str, source_file_hash: str) -> str:
    """Deterministic, stable id derived from the pack's content identity."""
    digest = hashlib.sha1(
        f"{pack_id}|{manifest_hash}|{source_file_hash}".encode("utf-8")
    ).hexdigest()
    return f"pdfregprop-{digest[:10]}"


def _is_eval_only_intent(intended_use: str) -> bool:
    """True when the declared intended use signals evaluation-only material."""
    low = intended_use.lower()
    return "eval" in low or "benchmark" in low or "test only" in low


def _lineage_note(manifest: dict) -> str:
    """A compact, human-readable provenance note for the proposed entry."""
    excluded = manifest.get("excluded_chunk_ids") or []
    return (
        "Imported PDF knowledge pack "
        f"(importer {manifest.get('importer_version', '')}). "
        f"approval_id={manifest.get('approval_id', '')}; "
        f"approved_by={manifest.get('approval_actor', '')}; "
        f"approved_at={manifest.get('approval_timestamp', '')}; "
        f"source_file_hash={manifest.get('source_file_hash', '')}; "
        f"preview_fingerprint={manifest.get('preview_fingerprint', '')}; "
        f"manifest_hash={manifest.get('manifest_hash', '')}; "
        f"intended_use={manifest.get('intended_use', '')}; "
        f"chunks={manifest.get('chunk_count', 0)}; "
        f"excluded={len(excluded)}. "
        "Provenance and permission are operator-declared; verify before "
        "activating."
    )


def _collect_findings(manifest: dict, *, existing_source_id_present: bool
                      ) -> List[RegistryProposalFinding]:
    """Derive deterministic review findings from the manifest. Pure."""
    findings: List[RegistryProposalFinding] = []

    authority = str(manifest.get("authority_level") or "").strip()
    if authority not in _KNOWN_AUTHORITIES:
        findings.append(RegistryProposalFinding(
            ProposalFindingCode.UNKNOWN_PACK_AUTHORITY,
            FindingSeverity.INFO,
            f"declared authority_level {authority!r} is not a recognised level; "
            f"proposed as {AuthorityLevel.UNKNOWN.value!r} for human review"))

    if not str(manifest.get("provenance") or "").strip():
        findings.append(RegistryProposalFinding(
            ProposalFindingCode.MISSING_PROVENANCE,
            FindingSeverity.WARNING,
            "manifest declares no provenance; a human should record where this "
            "PDF came from before the source is trusted"))

    if not str(manifest.get("permission_or_licence") or "").strip():
        findings.append(RegistryProposalFinding(
            ProposalFindingCode.MISSING_PERMISSION,
            FindingSeverity.WARNING,
            "manifest declares no permission/licence; a human should confirm "
            "the pack may be used before the source is trusted"))

    if _is_eval_only_intent(str(manifest.get("intended_use") or "")):
        findings.append(RegistryProposalFinding(
            ProposalFindingCode.EVAL_ONLY_INTENT,
            FindingSeverity.WARNING,
            "declared intended_use signals evaluation-only material; proposed "
            "as a draft so it is not activated as a trusted knowledge source"))

    unresolved = manifest.get("unresolved_nonblocking_findings") or []
    if unresolved:
        findings.append(RegistryProposalFinding(
            ProposalFindingCode.UNRESOLVED_IMPORT_FINDINGS,
            FindingSeverity.WARNING,
            f"import left {len(unresolved)} unresolved non-blocking finding(s); "
            "a human should review them before the source is trusted"))

    excluded = manifest.get("excluded_chunk_ids") or []
    if excluded:
        findings.append(RegistryProposalFinding(
            ProposalFindingCode.EXCLUDED_CHUNKS_PRESENT,
            FindingSeverity.INFO,
            f"{len(excluded)} chunk(s) were excluded at import; the registry "
            "entry describes a partial pack"))

    if existing_source_id_present:
        findings.append(RegistryProposalFinding(
            ProposalFindingCode.DUPLICATE_SOURCE_ID,
            FindingSeverity.ERROR,
            "a registry entry already uses this source_id; a human should "
            "decide whether this pack updates, supersedes, or renames it "
            "rather than adding a duplicate"))

    if not findings:
        findings.append(RegistryProposalFinding(
            ProposalFindingCode.COMPLETE_LINEAGE,
            FindingSeverity.INFO,
            "declared authority, provenance, and permission are present and the "
            "import is clean; proposed as an active source for human approval"))

    findings.sort(key=lambda f: (_SEVERITY_RANK[f.severity], f.code))
    return findings


def _decide(findings: List[RegistryProposalFinding]) -> str:
    """Worst-finding-wins recommendation. Pure."""
    codes = {f.code for f in findings}
    if ProposalFindingCode.DUPLICATE_SOURCE_ID in codes:
        return ProposalDecision.CONFLICT_EXISTING_SOURCE
    if any(f.severity is FindingSeverity.WARNING for f in findings):
        return ProposalDecision.PROPOSE_NEW_DRAFT
    return ProposalDecision.PROPOSE_NEW_ACTIVE


def _proposed_status(decision: str) -> SourceStatus:
    """Active only for a clean, conflict-free pack; otherwise draft."""
    if decision == ProposalDecision.PROPOSE_NEW_ACTIVE:
        return SourceStatus.ACTIVE
    return SourceStatus.DRAFT


def propose_registry_entry_from_manifest(
    manifest: dict, *,
    existing_entries: Optional[List[SourceRegistryEntry]] = None,
    now: Optional[datetime] = None,
) -> PdfPackRegistryProposal:
    """Propose a source-registry entry from an imported pack manifest. Pure.

    Reads the manifest dict (and, optionally, the existing registry entries for a
    read-only duplicate-id check) and returns a :class:`PdfPackRegistryProposal`.
    It writes nothing and changes no durable state. The proposed entry is
    ``active`` only when the pack's declared lineage is complete and there is no
    id conflict; otherwise it is proposed as a ``draft`` for human ratification.
    """
    pack_id = str(manifest.get("pack_id") or "").strip()
    if not pack_id:
        raise ValueError("manifest has no pack_id; cannot propose a registry entry")

    created_at = (now or datetime.now(timezone.utc)).isoformat()

    existing_index = index_by_id(existing_entries) if existing_entries else {}
    duplicate = pack_id in existing_index

    findings = _collect_findings(
        manifest, existing_source_id_present=duplicate)
    decision = _decide(findings)
    status = _proposed_status(decision)

    authority_raw = str(manifest.get("authority_level") or "").strip()
    authority = (AuthorityLevel(authority_raw)
                 if authority_raw in _KNOWN_AUTHORITIES
                 else AuthorityLevel.UNKNOWN)

    proposed_entry = SourceRegistryEntry(
        source_id=pack_id,
        title=str(manifest.get("name") or pack_id),
        source_type=PROPOSED_SOURCE_TYPE,
        authority_level=authority,
        owner=str(manifest.get("approval_actor") or ""),
        created_at=manifest.get("import_timestamp"),
        last_reviewed_at=manifest.get("approval_timestamp"),
        freshness_policy="static",
        stale_after_days=None,
        topics=[],
        linked_decisions=[],
        supersedes=[],
        superseded_by=[],
        status=status,
        notes=_lineage_note(manifest),
    )

    return PdfPackRegistryProposal(
        proposal_id=_proposal_id(
            pack_id,
            str(manifest.get("manifest_hash") or ""),
            str(manifest.get("source_file_hash") or "")),
        pack_id=pack_id,
        decision=decision,
        proposed_entry=proposed_entry,
        findings=findings,
        manifest_hash=str(manifest.get("manifest_hash") or ""),
        source_file_hash=str(manifest.get("source_file_hash") or ""),
        preview_fingerprint=str(manifest.get("preview_fingerprint") or ""),
        approval_id=str(manifest.get("approval_id") or ""),
        created_at=created_at,
    )


def propose_registry_entry_from_pack(
    pack_path: str | Path, *,
    registry_path: Optional[str | Path] = None,
    now: Optional[datetime] = None,
) -> PdfPackRegistryProposal:
    """Convenience: load the manifest (and optional registry) then propose. Pure.

    ``pack_path`` may be a pack directory or a manifest file. When
    ``registry_path`` is given, the existing registry is loaded read-only for a
    duplicate-id check; the registry is never modified.
    """
    manifest = load_pack_manifest(pack_path)
    existing = load_registry(registry_path) if registry_path else None
    return propose_registry_entry_from_manifest(
        manifest, existing_entries=existing, now=now)


# -----------------------------------------------------------------------------
# Rendering / export
# -----------------------------------------------------------------------------

def render_registry_proposal_markdown(proposal: PdfPackRegistryProposal) -> str:
    """Render a deterministic Markdown view of a proposal. No side effects."""
    entry = proposal.proposed_entry
    lines: List[str] = []
    lines.append(f"# Imported PDF pack -> source registry proposal "
                 f"({PROPOSAL_LAYER_VERSION}; proposal-only)")
    lines.append("")
    lines.append("A **proposal**, not an applied change: it requires human "
                 "approval and writes nothing to the registry. Provenance and "
                 "permission are operator-declared; a pack with gaps is proposed "
                 "as a draft, not an active source.")
    lines.append("")
    lines.append(f"- proposal_id: {proposal.proposal_id}")
    lines.append(f"- pack_id: {proposal.pack_id}")
    lines.append(f"- decision: {proposal.decision}")
    lines.append(f"- proposed status: {entry.status.value}")
    lines.append(f"- approval required: "
                 f"{'yes' if proposal.requires_human_approval else 'no'} "
                 f"(status: {proposal.status})")
    lines.append("")

    lines.append("## Proposed registry entry")
    lines.append("")
    lines.append(f"- source_id: {entry.source_id}")
    lines.append(f"- title: {entry.title}")
    lines.append(f"- source_type: {entry.source_type}")
    lines.append(f"- authority_level: {entry.authority_level.value}")
    lines.append(f"- owner: {entry.owner}")
    lines.append(f"- created_at: {entry.created_at}")
    lines.append(f"- last_reviewed_at: {entry.last_reviewed_at}")
    lines.append(f"- status: {entry.status.value}")
    lines.append("")

    lines.append("## Content lineage")
    lines.append("")
    lines.append(f"- manifest_hash: {proposal.manifest_hash}")
    lines.append(f"- source_file_hash: {proposal.source_file_hash}")
    lines.append(f"- preview_fingerprint: {proposal.preview_fingerprint}")
    lines.append(f"- approval_id: {proposal.approval_id}")
    lines.append("")

    lines.append("## Review findings")
    lines.append("")
    lines.append("| severity | code | message |")
    lines.append("|---|---|---|")
    for f in proposal.findings:
        lines.append(f"| {f.severity.value} | {f.code} | {f.message} |")
    lines.append("")

    lines.append("## Ready-to-paste registry record")
    lines.append("")
    lines.append("After approval, a human may paste this JSONL line into the "
                 "registry (the registry is never written automatically):")
    lines.append("")
    lines.append("```json")
    lines.append(entry.to_json())
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def registry_proposal_to_json(proposal: PdfPackRegistryProposal) -> str:
    """Serialise a proposal to deterministic pretty JSON."""
    return json.dumps(proposal.to_dict(), indent=2, sort_keys=True,
                      ensure_ascii=False)


def write_registry_proposal(proposal: PdfPackRegistryProposal,
                            path: str | Path) -> None:
    """Write a proposal to a JSON file. The ONLY writer in this layer.

    This export writer touches *only* the given path. It never writes the source
    registry (``save_registry`` remains the registry's sole writer and is not
    called here), the pack, the knowledge library, a memory ledger, or any
    review queue.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        registry_proposal_to_json(proposal) + "\n", encoding="utf-8")
