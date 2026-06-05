"""Tests for the v6.9 governed Hugging Face import approval contracts (Phase A).

These cover the fail-closed binding of an :class:`HFDatasetApproval` to an exact
dataset id, revision, and metadata fingerprint, the eval/knowledge scope split,
expiry, and licence/provenance drift. Approval is never produced automatically;
the lifecycle only validates a human-authored approval against a concrete
request and the live card.
"""
from __future__ import annotations

import ast
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import hf_import_lifecycle as life  # noqa: E402
from agent.hf_data_adapter import (  # noqa: E402
    HuggingFaceDatasetMetadata,
    assess_hf_metadata,
)
from agent.hf_import_lifecycle import (  # noqa: E402
    HFApprovalScope,
    HFDatasetApproval,
    HFImportIntent,
    HFImportRequest,
    HFValidationCode,
    compute_assessment_fingerprint,
    compute_metadata_fingerprint,
)

_DEMO_META = ROOT / "demos" / "hf_dataset_metadata_example.json"
_DEMO_APPROVAL = ROOT / "demos" / "hf_dataset_approval_example.json"
_NOW = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)


def _metadata() -> HuggingFaceDatasetMetadata:
    return HuggingFaceDatasetMetadata.from_dict(
        json.loads(_DEMO_META.read_text(encoding="utf-8")))


def _assessment():
    return assess_hf_metadata(_metadata()).assessment


def _approval(**overrides) -> HFDatasetApproval:
    meta = _metadata()
    assessment = _assessment()
    base = dict(
        approval_id="hfappr-test",
        dataset_id="demo/governed-qa",
        dataset_revision="main",
        metadata_fingerprint=compute_metadata_fingerprint(
            meta, assessment=assessment),
        intake_assessment_fingerprint=compute_assessment_fingerprint(assessment),
        approved_by="reviewer@local",
        approved_at="2026-05-02T09:00:00+00:00",
        approval_scope=HFApprovalScope.EVAL_AND_KNOWLEDGE,
        approved_split_names=("train",),
        approved_columns=("id", "question", "answer", "context"),
        row_limit=100,
        streaming_allowed=True,
        approved_intended_use="eval and knowledge",
        licence_snapshot="apache-2.0",
        provenance_snapshot="huggingface:demo/governed-qa (cited, homepage)",
    )
    base.update(overrides)
    return HFDatasetApproval(**base)


def _request(**overrides) -> HFImportRequest:
    base = dict(
        dataset_id="demo/governed-qa",
        dataset_revision="main",
        split="train",
        intent=HFImportIntent.KNOWLEDGE,
        requested_columns=("id", "question", "answer", "context"),
        requested_row_limit=50,
        streaming=True,
    )
    base.update(overrides)
    return HFImportRequest(**base)


def _codes(validation):
    return {c for c in validation.codes}


# 1. A matching request validates. -------------------------------------------


def test_matching_request_is_valid():
    v = life.validate_import_request(
        _approval(), _request(), metadata=_metadata(),
        assessment=_assessment(), now=_NOW)
    assert v.valid is True
    assert v.codes == (HFValidationCode.APPROVAL_VALID,)


# 2. Approval binds to dataset id. -------------------------------------------


def test_dataset_id_mismatch_fails_closed():
    v = life.validate_import_request(
        _approval(), _request(dataset_id="other/dataset"), now=_NOW)
    assert v.valid is False
    assert HFValidationCode.DATASET_ID_MISMATCH in _codes(v)


# 3. Approval binds to revision. ---------------------------------------------


def test_revision_mismatch_fails_closed():
    v = life.validate_import_request(
        _approval(), _request(dataset_revision="v2"), now=_NOW)
    assert v.valid is False
    assert HFValidationCode.REVISION_MISMATCH in _codes(v)


# 4. Approval binds to the exact metadata fingerprint. -----------------------


def test_metadata_fingerprint_mismatch_fails_closed():
    drifted = _approval(metadata_fingerprint="hfmeta-deadbeefdeadbeef")
    v = life.validate_import_request(
        drifted, _request(), metadata=_metadata(), assessment=_assessment(),
        now=_NOW)
    assert v.valid is False
    assert HFValidationCode.METADATA_FINGERPRINT_MISMATCH in _codes(v)


# 5. Eval-only approval cannot authorise a knowledge import. -----------------


def test_eval_only_approval_rejects_knowledge_import():
    eval_only = _approval(approval_scope=HFApprovalScope.EVAL_ONLY,
                          approved_intended_use="eval")
    v = life.validate_import_request(
        eval_only, _request(intent=HFImportIntent.KNOWLEDGE), now=_NOW)
    assert v.valid is False
    assert HFValidationCode.KNOWLEDGE_REQUIRES_KNOWLEDGE_APPROVAL in _codes(v)


# 6. Knowledge-only approval cannot authorise an eval import. ----------------


def test_knowledge_only_approval_rejects_eval_import():
    knowledge_only = _approval(approval_scope=HFApprovalScope.KNOWLEDGE_ONLY,
                               approved_intended_use="knowledge")
    v = life.validate_import_request(
        knowledge_only, _request(intent=HFImportIntent.EVAL), now=_NOW)
    assert v.valid is False
    assert HFValidationCode.EVAL_REQUIRES_EVAL_APPROVAL in _codes(v)


# 7. An expired approval fails closed. ---------------------------------------


def test_expired_approval_fails_closed():
    expired = _approval(expires_at="2026-01-01T00:00:00+00:00")
    v = life.validate_import_request(expired, _request(), now=_NOW)
    assert v.valid is False
    assert HFValidationCode.APPROVAL_EXPIRED in _codes(v)


# 8. A changed licence invalidates the approval. -----------------------------


def test_changed_licence_invalidates_approval():
    v = life.validate_import_request(
        _approval(), _request(licence_snapshot="cc-by-nc-4.0"), now=_NOW)
    assert v.valid is False
    assert HFValidationCode.LICENCE_CHANGED in _codes(v)


# 9. A changed provenance invalidates the approval. --------------------------


def test_changed_provenance_invalidates_approval():
    v = life.validate_import_request(
        _approval(), _request(provenance_snapshot="unknown"), now=_NOW)
    assert v.valid is False
    assert HFValidationCode.PROVENANCE_CHANGED in _codes(v)


# 10. An increased row cap fails without a new approval. ---------------------


def test_increased_row_cap_fails_closed():
    v = life.validate_import_request(
        _approval(), _request(requested_row_limit=500), now=_NOW)
    assert v.valid is False
    assert HFValidationCode.ROW_LIMIT_EXCEEDED in _codes(v)


# 11. Columns outside the approved set fail closed. --------------------------


def test_unapproved_columns_fail_closed():
    v = life.validate_import_request(
        _approval(), _request(requested_columns=("id", "secret_field")),
        now=_NOW)
    assert v.valid is False
    assert HFValidationCode.COLUMNS_NOT_APPROVED in _codes(v)


# 12. A split outside the approved set fails closed. -------------------------


def test_unapproved_split_fails_closed():
    v = life.validate_import_request(
        _approval(), _request(split="test"), now=_NOW)
    assert v.valid is False
    assert HFValidationCode.SPLIT_NOT_APPROVED in _codes(v)


# 13. A non-approval intake decision blocks import. --------------------------


def test_non_approval_intake_decision_blocks_import():
    # An unlicensed card assesses to needs_review, not an approval.
    unlicensed = HuggingFaceDatasetMetadata.from_dict({
        "id": "demo/governed-qa",
        "card_data": {"language": ["en"]},
        "homepage": "https://huggingface.co/datasets/demo/governed-qa",
    })
    assessment = assess_hf_metadata(unlicensed).assessment
    v = life.validate_import_request(
        _approval(intake_assessment_fingerprint=""), _request(),
        assessment=assessment, now=_NOW)
    assert v.valid is False
    assert HFValidationCode.INTAKE_NOT_APPROVED in _codes(v)


# 14. Eval-only intake decision cannot back a knowledge import. --------------


def test_eval_intake_decision_cannot_back_knowledge_import():
    # A synthetic card is approved for eval only.
    synthetic = HuggingFaceDatasetMetadata.from_dict({
        "id": "demo/governed-qa",
        "card_data": {"license": "apache-2.0", "language": ["en"]},
        "tags": ["synthetic"],
        "homepage": "https://huggingface.co/datasets/demo/governed-qa",
    })
    assessment = assess_hf_metadata(synthetic).assessment
    v = life.validate_import_request(
        _approval(intake_assessment_fingerprint=""),
        _request(intent=HFImportIntent.KNOWLEDGE),
        assessment=assessment, now=_NOW)
    assert v.valid is False
    assert HFValidationCode.INTAKE_NOT_APPROVED in _codes(v)


# 15. The fingerprint is deterministic and excludes volatile counters. -------


def test_metadata_fingerprint_is_deterministic_and_governance_bound():
    meta = _metadata()
    first = compute_metadata_fingerprint(meta)
    second = compute_metadata_fingerprint(meta)
    assert first == second
    assert first.startswith("hfmeta-")
    # Popularity counters do not change the fingerprint.
    noisier = HuggingFaceDatasetMetadata.from_dict({
        **json.loads(_DEMO_META.read_text(encoding="utf-8")),
        "downloads": 999999, "likes": 4242,
    })
    assert compute_metadata_fingerprint(noisier) == first
    # A licence change does change it.
    relicensed = HuggingFaceDatasetMetadata.from_dict({
        **json.loads(_DEMO_META.read_text(encoding="utf-8")),
        "card_data": {"license": "cc-by-nc-4.0"},
    })
    assert compute_metadata_fingerprint(relicensed) != first


# 16. The demo approval file loads and matches the demo card. ----------------


def test_demo_approval_matches_demo_card():
    approval = life.load_approval(_DEMO_APPROVAL)
    meta = _metadata()
    assessment = _assessment()
    assert approval.metadata_fingerprint == compute_metadata_fingerprint(
        meta, assessment=assessment)
    assert approval.intake_assessment_fingerprint == \
        compute_assessment_fingerprint(assessment)
    v = life.validate_import_request(
        approval, _request(), metadata=meta, assessment=assessment, now=_NOW)
    assert v.valid is True


# 17. Approval round-trips through from_dict/to_dict. ------------------------


def test_approval_round_trips():
    approval = _approval()
    reloaded = HFDatasetApproval.from_dict(approval.to_dict())
    assert reloaded == approval
    assert approval.to_dict()["_record"] == "hf_dataset_approval"


# 18. A malformed approval (missing required field) fails closed. ------------


def test_malformed_approval_rejected():
    payload = _approval().to_dict()
    payload.pop("dataset_revision")
    with pytest.raises(ValueError):
        HFDatasetApproval.from_dict(payload)


# 19. The validation record serialises deterministically. -------------------


def test_validation_serialisation_is_deterministic():
    a = life.validate_import_request(_approval(), _request(), now=_NOW)
    b = life.validate_import_request(_approval(), _request(), now=_NOW)
    assert a.to_dict() == b.to_dict()
    assert a.to_dict()["_record"] == "hf_import_validation"


# 20. The approval module imports no forbidden writers/clients. --------------


def test_module_imports_no_forbidden_dependencies():
    source = Path(life.__file__).read_text(encoding="utf-8")
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
        "agent.memory_ledger", "agent.source_registry", "agent.knowledge_library",
        "agent.hf_dataset_importer", "agent.project_packs",
        "datasets", "huggingface_hub", "requests", "urllib", "urllib.request",
        "httpx", "openai", "anthropic",
    }
    leaked = imported & forbidden
    assert not leaked, f"lifecycle must not import: {leaked}"
