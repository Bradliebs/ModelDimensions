"""v7.2 regression review queue: item identity, fingerprints and import purity.

Pins that a review item binds to the exact monitoring evidence it was created
against, that identity is deterministic (a duplicate import is the same item),
and that the module imports no lifecycle executor, ledger/registry writer,
proposal applier, retrieval/pack writer or LLM client.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import agent.regression_review_queue as rrq  # noqa: E402
from _regression_review_helpers import FIXED_NOW, run_dict  # noqa: E402


def test_import_builds_item_bound_to_monitoring_evidence():
    rd = run_dict()
    item = rrq.import_monitoring_recommendation(rd, created_at=FIXED_NOW)
    assert item is not None
    assert item.review_item_id.startswith(rrq.REVIEW_ITEM_PREFIX)
    assert item.monitoring_run_fingerprint == rd["record_hash"]
    assert item.monitoring_run_id == rd["monitoring_run_id"]
    assert item.active_state_hash == rd["snapshot"]["active_state_hash"]
    assert item.baseline_hash == rd["baseline_hash"]
    assert item.corpus_fingerprints == (rd["corpus_fingerprint"],)
    assert item.policy_fingerprint == rd["policy_fingerprint"]
    assert item.retrieval_config_fingerprint == rd["retrieval_config_fingerprint"]
    assert item.affected_pack_ids == ("p1",)
    assert item.affected_pack_fingerprints == ("packfp-a",)
    assert item.status == rrq.ReviewItemStatus.PENDING


def test_recommendation_fingerprint_is_deterministic_and_prefixed():
    rd = run_dict()
    a = rrq.import_monitoring_recommendation(rd, created_at=FIXED_NOW)
    b = rrq.import_monitoring_recommendation(rd, created_at="2025-06-06T00:00:00+00:00")
    assert a.recommendation_fingerprint == b.recommendation_fingerprint
    assert a.recommendation_fingerprint.startswith(rrq.RECOMMENDATION_FP_PREFIX)
    # identity ignores created_at — the same evidence is the same item
    assert a.review_item_id == b.review_item_id


def test_changed_evidence_changes_identity():
    base = rrq.import_monitoring_recommendation(run_dict(), created_at=FIXED_NOW)
    other_run = rrq.import_monitoring_recommendation(
        run_dict(corpus_fingerprint="moncorpus-DIFFERENT"), created_at=FIXED_NOW)
    assert base.review_item_id != other_run.review_item_id
    assert base.monitoring_run_fingerprint != other_run.monitoring_run_fingerprint


def test_proposed_action_type_maps_from_recommendation():
    cases = {
        rrq.MonitoringRecommendationCode.DEACTIVATE_RECOMMENDED:
            rrq.ActionRequestType.REQUEST_DEACTIVATION,
        rrq.MonitoringRecommendationCode.ROLLBACK_RECOMMENDED:
            rrq.ActionRequestType.REQUEST_ROLLBACK,
        rrq.MonitoringRecommendationCode.BLOCK_FUTURE_ACTIVATION:
            rrq.ActionRequestType.REQUEST_ACTIVATION_BLOCK,
        rrq.MonitoringRecommendationCode.INVESTIGATE:
            rrq.ActionRequestType.REQUEST_INVESTIGATION,
    }
    for rec, action in cases.items():
        rd = run_dict(recommendation=rec)
        item = rrq.import_monitoring_recommendation(rd, created_at=FIXED_NOW)
        assert item.proposed_action_type == action


def test_severity_is_info_when_no_findings():
    rd = run_dict(severity=None)  # a run that carries no regression findings
    item = rrq.import_monitoring_recommendation(
        rd, recommendation=rrq.MonitoringRecommendationCode.INVESTIGATE.value,
        created_at=FIXED_NOW)
    assert item.severity == "info"


def test_findings_carry_no_retrieved_text():
    item = rrq.import_monitoring_recommendation(run_dict(), created_at=FIXED_NOW)
    for f in item.regression_findings:
        assert set(f) <= {"finding_code", "severity", "case_ids",
                          "confidence", "candidate_pack_ids"}


def test_roundtrip_to_and_from_dict():
    item = rrq.import_monitoring_recommendation(run_dict(), created_at=FIXED_NOW)
    again = rrq.RegressionReviewItem.from_dict(item.to_dict())
    assert again == item


def test_non_monitoring_record_is_rejected():
    try:
        rrq.import_monitoring_recommendation({"_record": "something_else"})
    except ValueError:
        return
    raise AssertionError("expected ValueError for a non-monitoring record")


def test_module_imports_no_executor_writer_network_or_llm():
    source = Path(rrq.__file__).read_text(encoding="utf-8")
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
        "ActivationStateManager", "MemoryLedger", "WorkbenchService",
        "PackRegistry", "save_registry", "propose_source_updates",
        "build_memory_proposals", "apply_review_to_queue",
        "agent.workbench_service", "agent.memory_ledger", "agent.source_registry",
        "agent.memory_proposal_quality", "agent.pack_builder",
        "agent.project_packs", "agent.hf_knowledge_pack_importer",
        "agent.pdf_pack_importer", "datasets", "huggingface_hub", "requests",
        "urllib", "urllib.request", "httpx", "openai", "anthropic",
    }
    leaked = imported & forbidden
    assert not leaked, f"review queue must not import: {leaked}"

    body = source.replace(ast.get_docstring(tree, clean=False) or "", "", 1)
    for token in (
            "ActivationStateManager(", "MemoryLedger(", "save_registry(",
            "propose_source", "build_memory_proposals(", "load_dataset",
            "hf_hub_download", "requests.get", "urlopen", "OpenAI(",
            "Anthropic(", ".activate(", ".deactivate(", ".rollback(",
            ".supersede("):
        assert token not in body, f"review queue must not reference {token!r}"


def test_module_only_imports_stdlib_and_readonly_contracts():
    source = Path(rrq.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    roots = set()
    activation_names = set()
    monitor_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
            if node.module == "agent.knowledge_pack_activation":
                activation_names.update(a.name for a in node.names)
            if node.module == "agent.active_pack_monitor":
                monitor_names.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
    assert roots <= {
        "__future__", "hashlib", "json", "os", "tempfile", "dataclasses",
        "datetime", "pathlib", "typing", "agent",
    }, f"unexpected import roots: {roots}"
    assert "ActivationStateManager" not in activation_names
    assert activation_names <= {
        "ActivationStateManifest", "ActivePackState", "PackLifecycleState"}
    assert monitor_names <= {
        "ConfidenceBand", "MONITOR_LAYER_VERSION", "MonitoringRecommendationCode"}
