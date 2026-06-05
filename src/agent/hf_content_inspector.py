"""Governed Hugging Face content inspection — Phase C (v6.9).

Phase C normalizes a sampled row into canonical fields; this module inspects
those fields for **personal data and unsafe content** before any import phase
runs. It is a deterministic, regex-based screen — *not* a statistical classifier
and *not* a guarantee of safety — and it is fail-closed: when it is unsure, it
flags.

Two policy levers, both defaulting to the strict setting:

* ``pii_policy="block_knowledge"`` — a PII finding caps a row at *eval* use and
  blocks it from *knowledge* import (mirroring the v6.2A intake rule that
  personal data can never auto-classify as ``approved_for_knowledge``).
* ``content_policy="block_on_unsafe"`` — an unsafe-content finding blocks a row
  from *both* eval and knowledge import.

Governance constraints honoured here:

* Findings **redact** the matched text — only the finding code, the field name,
  and a match count are recorded. Raw PII never lands in a finding, a report, or
  a rendered summary.
* This module imports no writer: it cannot touch the ``MemoryLedger``, the source
  registry, a proposal, a retrieval index, an eval/knowledge pack, or an LLM.

The only durable write is :func:`write_content_inspection_report`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

from agent.hf_row_normalizer import HFNormalizedRow

INSPECTOR_VERSION = "hf-content-inspector-v6.9"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Deterministic detectors
# --------------------------------------------------------------------------- #


class HFContentFindingCode(str, Enum):
    """Stable, auditable reason codes for content findings."""

    EMAIL_ADDRESS = "email_address"
    PHONE_NUMBER = "phone_number"
    US_SSN = "us_ssn"
    CREDIT_CARD_LIKE = "credit_card_like"
    IP_ADDRESS = "ip_address"
    POSSIBLE_UNSAFE_CONTENT = "possible_unsafe_content"


class HFContentSeverity(str, Enum):
    """How strongly a content finding constrains import."""

    UNSAFE_BLOCK = "unsafe_block"  # blocks eval AND knowledge
    PII_KNOWLEDGE_BLOCK = "pii_knowledge_block"  # caps at eval, blocks knowledge
    INFO = "info"


_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_US_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CREDIT_CARD_RE = re.compile(r"\b(?:\d[ \-]?){13,16}\b")
_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?\d{1,3}[ \-.]?)?(?:\(\d{3}\)|\d{3})[ \-.]\d{3}[ \-.]\d{4}(?!\d)")
_IPV4_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")

# A small, deterministic deny-list. This is a deliberately conservative
# placeholder screen, not a content-safety classifier; it exists so that the
# governed path fails closed on obviously unsafe markers rather than silently
# importing them.
_UNSAFE_MARKERS: Tuple[str, ...] = (
    "build a bomb", "make a bomb", "how to kill", "child sexual",
    "credit card dump", "ransomware payload", "synthesize a nerve agent",
)

_PII_DETECTORS: Tuple[Tuple[HFContentFindingCode, "re.Pattern[str]"], ...] = (
    (HFContentFindingCode.EMAIL_ADDRESS, _EMAIL_RE),
    (HFContentFindingCode.US_SSN, _US_SSN_RE),
    (HFContentFindingCode.CREDIT_CARD_LIKE, _CREDIT_CARD_RE),
    (HFContentFindingCode.PHONE_NUMBER, _PHONE_RE),
    (HFContentFindingCode.IP_ADDRESS, _IPV4_RE),
)

# IP address is weaker evidence of personal data; keep it advisory.
_INFO_ONLY_PII = frozenset({HFContentFindingCode.IP_ADDRESS})


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HFContentPolicy:
    """How PII and unsafe-content findings constrain import (fail-closed)."""

    pii_policy: str = "block_knowledge"
    content_policy: str = "block_on_unsafe"

    @property
    def pii_blocks_knowledge(self) -> bool:
        return self.pii_policy != "allow"

    @property
    def unsafe_blocks_all(self) -> bool:
        return self.content_policy != "allow"

    @classmethod
    def from_approval(cls, approval) -> "HFContentPolicy":
        return cls(
            pii_policy=str(getattr(approval, "pii_policy", "block_knowledge")),
            content_policy=str(getattr(approval, "content_policy",
                                       "block_on_unsafe")),
        )

    def to_dict(self) -> dict:
        return {
            "pii_policy": self.pii_policy,
            "content_policy": self.content_policy,
        }


# --------------------------------------------------------------------------- #
# Findings
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HFContentFinding:
    """A single redacted content observation about one field of a row."""

    code: HFContentFindingCode
    severity: HFContentSeverity
    field_name: str
    match_count: int
    message: str

    @property
    def is_pii(self) -> bool:
        return self.code in {c for c, _ in _PII_DETECTORS}

    @property
    def is_unsafe(self) -> bool:
        return self.code is HFContentFindingCode.POSSIBLE_UNSAFE_CONTENT

    def to_dict(self) -> dict:
        return {
            "code": self.code.value,
            "severity": self.severity.value,
            "field": self.field_name,
            "match_count": self.match_count,
            "message": self.message,
        }


@dataclass(frozen=True)
class HFContentInspection:
    """The fail-closed content verdict for one normalized row."""

    row_id: str
    findings: Tuple[HFContentFinding, ...]
    pii_detected: bool
    unsafe_detected: bool
    blocks_knowledge: bool
    blocks_eval: bool
    inspected_at: str
    inspector_version: str = INSPECTOR_VERSION

    @property
    def ok_for_eval(self) -> bool:
        return not self.blocks_eval

    @property
    def ok_for_knowledge(self) -> bool:
        return not self.blocks_knowledge

    def to_dict(self) -> dict:
        return {
            "_record": "hf_content_inspection",
            "row_id": self.row_id,
            "pii_detected": self.pii_detected,
            "unsafe_detected": self.unsafe_detected,
            "blocks_knowledge": self.blocks_knowledge,
            "blocks_eval": self.blocks_eval,
            "inspected_at": self.inspected_at,
            "inspector_version": self.inspector_version,
            "findings": [f.to_dict() for f in self.findings],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


# --------------------------------------------------------------------------- #
# Inspection
# --------------------------------------------------------------------------- #


def _inspect_text(field_name: str, text: str) -> List[HFContentFinding]:
    """Return redacted findings for one field's text (match counts only)."""
    findings: List[HFContentFinding] = []
    for code, pattern in _PII_DETECTORS:
        count = len(pattern.findall(text))
        if count:
            severity = (HFContentSeverity.INFO if code in _INFO_ONLY_PII
                        else HFContentSeverity.PII_KNOWLEDGE_BLOCK)
            findings.append(HFContentFinding(
                code=code, severity=severity, field_name=field_name,
                match_count=count,
                message=f"{count} {code.value} match(es) in field {field_name!r}"))
    lowered = text.lower()
    unsafe_hits = sum(1 for marker in _UNSAFE_MARKERS if marker in lowered)
    if unsafe_hits:
        findings.append(HFContentFinding(
            code=HFContentFindingCode.POSSIBLE_UNSAFE_CONTENT,
            severity=HFContentSeverity.UNSAFE_BLOCK,
            field_name=field_name, match_count=unsafe_hits,
            message=f"{unsafe_hits} unsafe-marker match(es) in field "
                    f"{field_name!r}"))
    return findings


def inspect_normalized_row(
    row: HFNormalizedRow,
    *,
    policy: Optional[HFContentPolicy] = None,
    now: Optional[datetime] = None,
) -> HFContentInspection:
    """Inspect every normalized field of a row; fail closed per ``policy``."""
    policy = policy or HFContentPolicy()
    now = now or _utc_now()
    findings: List[HFContentFinding] = []
    for field_name, text in row.text_items():
        findings.extend(_inspect_text(field_name, text))

    pii_detected = any(f.is_pii and f.severity is not HFContentSeverity.INFO
                       for f in findings)
    unsafe_detected = any(f.is_unsafe for f in findings)
    blocks_eval = unsafe_detected and policy.unsafe_blocks_all
    blocks_knowledge = blocks_eval or (pii_detected and policy.pii_blocks_knowledge)

    return HFContentInspection(
        row_id=row.row_id,
        findings=tuple(findings),
        pii_detected=pii_detected,
        unsafe_detected=unsafe_detected,
        blocks_knowledge=blocks_knowledge,
        blocks_eval=blocks_eval,
        inspected_at=now.isoformat(),
    )


@dataclass(frozen=True)
class HFContentInspectionReport:
    """The auditable content verdict for a whole normalized sample."""

    dataset_id: str
    dataset_revision: str
    split: str
    inspections: Tuple[HFContentInspection, ...]
    policy: HFContentPolicy
    inspected_at: str
    inspector_version: str = INSPECTOR_VERSION

    @property
    def rows_with_pii(self) -> int:
        return sum(1 for i in self.inspections if i.pii_detected)

    @property
    def rows_with_unsafe(self) -> int:
        return sum(1 for i in self.inspections if i.unsafe_detected)

    @property
    def rows_blocked_from_knowledge(self) -> int:
        return sum(1 for i in self.inspections if i.blocks_knowledge)

    @property
    def rows_blocked_from_eval(self) -> int:
        return sum(1 for i in self.inspections if i.blocks_eval)

    def knowledge_safe_row_ids(self) -> Tuple[str, ...]:
        return tuple(i.row_id for i in self.inspections if not i.blocks_knowledge)

    def eval_safe_row_ids(self) -> Tuple[str, ...]:
        return tuple(i.row_id for i in self.inspections if not i.blocks_eval)

    def to_dict(self) -> dict:
        return {
            "_record": "hf_content_inspection_report",
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "split": self.split,
            "policy": self.policy.to_dict(),
            "row_count": len(self.inspections),
            "rows_with_pii": self.rows_with_pii,
            "rows_with_unsafe": self.rows_with_unsafe,
            "rows_blocked_from_knowledge": self.rows_blocked_from_knowledge,
            "rows_blocked_from_eval": self.rows_blocked_from_eval,
            "inspector_version": self.inspector_version,
            "inspected_at": self.inspected_at,
            "inspections": [i.to_dict() for i in self.inspections],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def inspect_normalized_rows(
    rows,
    *,
    dataset_id: str = "",
    dataset_revision: str = "",
    split: str = "",
    policy: Optional[HFContentPolicy] = None,
    now: Optional[datetime] = None,
) -> HFContentInspectionReport:
    """Inspect a sequence of normalized rows under one policy (read-only)."""
    policy = policy or HFContentPolicy()
    now = now or _utc_now()
    rows = tuple(rows)
    inspections = tuple(
        inspect_normalized_row(row, policy=policy, now=now) for row in rows)
    if rows:
        dataset_id = dataset_id or rows[0].dataset_id
        dataset_revision = dataset_revision or rows[0].dataset_revision
        split = split or rows[0].split
    return HFContentInspectionReport(
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        split=split,
        inspections=inspections,
        policy=policy,
        inspected_at=now.isoformat(),
    )


# --------------------------------------------------------------------------- #
# Rendering / writing
# --------------------------------------------------------------------------- #


def render_content_inspection_markdown(report: HFContentInspectionReport) -> str:
    """Deterministic summary of counts only. Never prints matched content."""
    lines = [
        "# Governed Hugging Face content inspection (v6.9; redacted)",
        "",
        f"- dataset: `{report.dataset_id}`",
        f"- revision: `{report.dataset_revision}`",
        f"- split: `{report.split}`",
        f"- pii policy: `{report.policy.pii_policy}`",
        f"- content policy: `{report.policy.content_policy}`",
        f"- rows inspected: {len(report.inspections)}",
        f"- rows with PII: {report.rows_with_pii}",
        f"- rows with unsafe content: {report.rows_with_unsafe}",
        f"- rows blocked from knowledge: {report.rows_blocked_from_knowledge}",
        f"- rows blocked from eval: {report.rows_blocked_from_eval}",
        f"- inspector: `{report.inspector_version}`",
    ]
    return "\n".join(lines) + "\n"


def write_content_inspection_report(report: HFContentInspectionReport, path) -> Path:
    """The only durable write: a single inspection-report JSON to ``path``."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.to_json() + "\n", encoding="utf-8")
    return out
