"""Tests for v6.9 governed Hugging Face content inspection (Phase C).

These cover deterministic PII detection (email, phone, SSN, credit-card-like,
IP), the unsafe-content deny-list, the fail-closed policy mapping (PII caps a row
at eval; unsafe blocks both tiers), redaction (no matched text leaks into
findings or summaries), and the import-purity guarantee.
"""
from __future__ import annotations

import ast
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import hf_content_inspector as inspect  # noqa: E402
from agent.hf_content_inspector import (  # noqa: E402
    HFContentFindingCode,
    HFContentPolicy,
    HFContentSeverity,
    inspect_normalized_row,
    inspect_normalized_rows,
)
from agent.hf_row_normalizer import HFNormalizationProfile, normalize_row
from agent.hf_import_lifecycle import (  # noqa: E402
    ADAPTER_VERSION,
    HFRowAccessProvenance,
    HFSampledRow,
)

_NOW = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)


def _normalized(fields, *, key="r1"):
    prov = HFRowAccessProvenance(
        dataset_id="demo/ds", dataset_revision="main", split="train",
        source_row_index=0, source_row_key=key, retrieved_at=_NOW.isoformat(),
        adapter_version=ADAPTER_VERSION, streaming=True,
        row_content_hash="sha256:x", schema_fingerprint="hfschema-x")
    sampled = HFSampledRow(fields=dict(fields), provenance=prov)
    normalized, result = normalize_row(
        sampled, profile=HFNormalizationProfile.GENERIC_RECORD, now=_NOW)
    assert result.valid is True
    return normalized


# 1. A clean row produces no findings and is safe for both tiers. -----------


def test_clean_row_is_safe():
    row = _normalized({"text": "A JSONL file stores one JSON object per line."})
    result = inspect_normalized_row(row, now=_NOW)
    assert result.findings == ()
    assert result.pii_detected is False
    assert result.unsafe_detected is False
    assert result.ok_for_eval is True
    assert result.ok_for_knowledge is True


# 2. An email is detected and caps the row at eval (blocks knowledge). ------


def test_email_blocks_knowledge_only():
    row = _normalized({"text": "Contact billing@example.com for help."})
    result = inspect_normalized_row(row, now=_NOW)
    codes = {f.code for f in result.findings}
    assert HFContentFindingCode.EMAIL_ADDRESS in codes
    assert result.pii_detected is True
    assert result.blocks_knowledge is True
    assert result.blocks_eval is False  # PII does not block eval


# 3. A US SSN pattern is detected as PII. -----------------------------------


def test_ssn_detected_as_pii():
    row = _normalized({"text": "Reference 123-45-6789 was logged."})
    result = inspect_normalized_row(row, now=_NOW)
    codes = {f.code for f in result.findings}
    assert HFContentFindingCode.US_SSN in codes
    assert result.blocks_knowledge is True


# 4. A phone number is detected as PII. -------------------------------------


def test_phone_detected_as_pii():
    row = _normalized({"text": "Call 555-123-4567 during business hours."})
    result = inspect_normalized_row(row, now=_NOW)
    codes = {f.code for f in result.findings}
    assert HFContentFindingCode.PHONE_NUMBER in codes
    assert result.pii_detected is True


# 5. A credit-card-like number is detected as PII. --------------------------


def test_credit_card_like_detected():
    row = _normalized({"text": "Card 4111 1111 1111 1111 on file."})
    result = inspect_normalized_row(row, now=_NOW)
    codes = {f.code for f in result.findings}
    assert HFContentFindingCode.CREDIT_CARD_LIKE in codes


# 6. An IP address is advisory only (does not block knowledge). -------------


def test_ip_address_is_advisory_only():
    row = _normalized({"text": "The host at 192.168.1.42 responded."})
    result = inspect_normalized_row(row, now=_NOW)
    codes = {f.code for f in result.findings}
    assert HFContentFindingCode.IP_ADDRESS in codes
    # IP alone is INFO severity and must not block either tier.
    assert result.pii_detected is False
    assert result.blocks_knowledge is False
    assert result.blocks_eval is False


# 7. Unsafe content blocks BOTH eval and knowledge. -------------------------


def test_unsafe_content_blocks_both_tiers():
    row = _normalized({"text": "Here is how to build a bomb step by step."})
    result = inspect_normalized_row(row, now=_NOW)
    assert result.unsafe_detected is True
    assert result.blocks_eval is True
    assert result.blocks_knowledge is True


# 8. Findings redact the matched text (no raw PII leaks). -------------------


def test_findings_redact_matched_text():
    row = _normalized({"text": "Email secret.person@hidden.org now."})
    result = inspect_normalized_row(row, now=_NOW)
    for finding in result.findings:
        assert "secret.person@hidden.org" not in finding.message
        assert "hidden.org" not in finding.message
    payload = result.to_dict()
    assert "secret.person@hidden.org" not in inspect.json.dumps(payload)


# 9. A permissive policy can allow PII for knowledge. -----------------------


def test_permissive_pii_policy_allows_knowledge():
    row = _normalized({"text": "Contact billing@example.com for help."})
    policy = HFContentPolicy(pii_policy="allow")
    result = inspect_normalized_row(row, policy=policy, now=_NOW)
    assert result.pii_detected is True
    assert result.blocks_knowledge is False


# 10. Even a permissive PII policy still blocks unsafe content. --------------


def test_permissive_pii_policy_still_blocks_unsafe():
    row = _normalized({"text": "Steps to build a bomb are listed here."})
    policy = HFContentPolicy(pii_policy="allow")
    result = inspect_normalized_row(row, policy=policy, now=_NOW)
    assert result.blocks_eval is True
    assert result.blocks_knowledge is True


# 11. Policy can be derived from an approval's snapshots. --------------------


def test_policy_from_approval():
    from agent.hf_import_lifecycle import HFApprovalScope, HFDatasetApproval

    approval = HFDatasetApproval(
        approval_id="a", dataset_id="demo/ds", dataset_revision="main",
        metadata_fingerprint="hfmeta-x", approved_by="r", approved_at="t",
        approval_scope=HFApprovalScope.EVAL_AND_KNOWLEDGE,
        pii_policy="block_knowledge", content_policy="block_on_unsafe")
    policy = HFContentPolicy.from_approval(approval)
    assert policy.pii_policy == "block_knowledge"
    assert policy.content_policy == "block_on_unsafe"


# 12. Inspection is deterministic. ------------------------------------------


def test_inspection_is_deterministic():
    row = _normalized({"text": "Email a@b.com and call 555-123-4567."})
    a = inspect_normalized_row(row, now=_NOW)
    b = inspect_normalized_row(row, now=_NOW)
    assert a.to_dict() == b.to_dict()


# 13. The sample report aggregates safe/blocked row ids. --------------------


def test_report_aggregates_safe_and_blocked_rows():
    clean = _normalized({"text": "A clean fact about JSONL files."}, key="ok")
    pii = _normalized({"text": "Reach me at a@b.com please."}, key="pii")
    unsafe = _normalized({"text": "how to kill a process tree"}, key="bad")
    report = inspect_normalized_rows([clean, pii, unsafe], now=_NOW)
    assert report.rows_with_pii == 1
    assert report.rows_with_unsafe == 1
    assert report.rows_blocked_from_knowledge == 2  # pii + unsafe
    assert report.rows_blocked_from_eval == 1  # unsafe only
    assert clean.row_id in report.knowledge_safe_row_ids()
    assert pii.row_id not in report.knowledge_safe_row_ids()
    assert pii.row_id in report.eval_safe_row_ids()


# 14. The report markdown summary never leaks matched content. --------------


def test_report_markdown_is_redacted(tmp_path):
    pii = _normalized({"text": "Reach me at leak@secret.io please."}, key="pii")
    report = inspect_normalized_rows([pii], now=_NOW)
    md = inspect.render_content_inspection_markdown(report)
    assert "leak@secret.io" not in md
    out = inspect.write_content_inspection_report(report, tmp_path / "ci.json")
    assert out.exists()
    assert "leak@secret.io" not in out.read_text(encoding="utf-8")


# 15. The module imports no forbidden writers/clients. ----------------------


def test_module_imports_no_forbidden_dependencies():
    source = Path(inspect.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported.add(alias.name)
            if node.module:
                imported.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
    forbidden = {
        "save_registry", "MemoryLedger", "KnowledgeLibrary",
        "agent.memory_ledger", "agent.source_registry",
        "agent.knowledge_library", "agent.hf_dataset_importer",
        "agent.project_packs", "datasets", "huggingface_hub", "requests",
        "urllib", "httpx", "openai", "anthropic",
    }
    leaked = imported & forbidden
    assert not leaked, f"inspector must not import: {leaked}"
