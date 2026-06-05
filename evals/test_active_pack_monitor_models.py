"""v7.1 active-pack monitoring: identities, fingerprints and import purity.

These pin the deterministic identity layer: snapshot/corpus/retrieval/baseline/run
fingerprints are stable for fixed inputs and change when the bound inputs change;
a stored run is invalidated when the active set, corpus, retrieval config or policy
changes; and the module imports only the standard library plus read-only names
from the activation contracts (no writer, no LLM, no network).
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import agent.active_pack_monitor as apm  # noqa: E402
from agent.knowledge_pack_activation import (  # noqa: E402
    ActivationStateManifest,
    ActivePackState,
    PackLifecycleState,
)

FIXED_NOW = "2024-01-01T00:00:00+00:00"


def _state() -> ActivationStateManifest:
    rec = ActivePackState(
        pack_id="p1", pack_version="1.0", pack_fingerprint="packfp-a",
        source_type="hf", source_id="src1", source_revision="r1",
        status=PackLifecycleState.ACTIVE, activated_at=FIXED_NOW,
        activation_approval_id="ap1", evaluation_report_id="ev1")
    return ActivationStateManifest(records=(rec,))


def _cases(n: int = 4):
    return [apm.MonitoringCase(case_id=f"c{i}", query=f"q{i}",
                              expected_sources=("src1",), critical_case=(i == 0))
            for i in range(n)]


def test_snapshot_fingerprint_is_deterministic_and_binds_active_set():
    snap = apm.snapshot_from_state(_state(), captured_at=FIXED_NOW)
    again = apm.snapshot_from_state(_state(), captured_at=FIXED_NOW)
    assert snap.snapshot_id == again.snapshot_id
    assert snap.snapshot_id.startswith(apm.SNAPSHOT_FP_PREFIX)
    assert snap.active_pack_ids == ("p1",)


def test_corpus_fingerprint_is_order_independent_and_changes_on_edit():
    cases = _cases()
    fp = apm.compute_corpus_fingerprint(cases)
    assert fp == apm.compute_corpus_fingerprint(list(reversed(cases)))
    edited = cases[:-1] + [apm.MonitoringCase(case_id="c9", query="different",
                                              expected_sources=("src1",))]
    assert apm.compute_corpus_fingerprint(edited) != fp


def test_retrieval_config_fingerprint_changes_with_backend():
    a = apm.compute_retrieval_config_fingerprint(backend="lexical")
    b = apm.compute_retrieval_config_fingerprint(backend="hybrid")
    assert a != b
    assert a.startswith(apm.RETRIEVAL_CFG_PREFIX)


def test_case_round_trips_through_dict():
    case = _cases(1)[0]
    restored = apm.MonitoringCase.from_dict(case.to_dict())
    assert restored.canonical() == case.canonical()


def _baseline_and_run():
    snap = apm.snapshot_from_state(_state(), captured_at=FIXED_NOW)
    cases = _cases()
    corpus_fp = apm.compute_corpus_fingerprint(cases)
    rcfg = apm.compute_retrieval_config_fingerprint(backend="lexical")
    results = [apm.MonitoringCaseResult(
        case_id=f"c{i}", case_class=apm.MonitoringCaseClass.PACK_RELEVANT,
        severity=apm.MonitoringSeverity.CRITICAL if i == 0
        else apm.MonitoringSeverity.MEDIUM,
        critical_case=(i == 0), hit=True, expected_source_recall=1.0,
        wrong_source_rate=0.0, passed=True) for i in range(4)]
    baseline = apm.create_baseline(
        baseline_id="b1", baseline_type=apm.BaselineType.PRE_ACTIVATION,
        snapshot=snap, corpus_fingerprint=corpus_fp,
        retrieval_config_fingerprint=rcfg, results=results, created_at=FIXED_NOW)
    run = apm.run_monitoring(
        baseline=baseline, snapshot=snap, results=results,
        corpus_fingerprint=corpus_fp, retrieval_config_fingerprint=rcfg,
        created_at=FIXED_NOW)
    return run, corpus_fp, rcfg


def test_run_record_hash_is_deterministic():
    run_a, _, _ = _baseline_and_run()
    run_b, _, _ = _baseline_and_run()
    assert run_a.record_hash == run_b.record_hash
    assert run_a.record_hash.startswith(apm.RECORD_HASH_PREFIX)


def test_run_is_invalidated_when_bound_inputs_change():
    run, corpus_fp, rcfg = _baseline_and_run()
    pol_fp = apm.DEFAULT_MONITORING_POLICY.policy_fingerprint
    assert run.is_valid_for(
        active_state_hash=run.snapshot.active_state_hash,
        corpus_fingerprint=corpus_fp, retrieval_config_fingerprint=rcfg,
        policy_fingerprint=pol_fp)
    assert not run.is_valid_for(
        active_state_hash="different", corpus_fingerprint=corpus_fp,
        retrieval_config_fingerprint=rcfg, policy_fingerprint=pol_fp)
    assert not run.is_valid_for(
        active_state_hash=run.snapshot.active_state_hash,
        corpus_fingerprint="moncorpus-other",
        retrieval_config_fingerprint=rcfg, policy_fingerprint=pol_fp)
    assert not run.is_valid_for(
        active_state_hash=run.snapshot.active_state_hash,
        corpus_fingerprint=corpus_fp, retrieval_config_fingerprint=rcfg,
        policy_fingerprint="different")


def test_module_imports_no_writer_or_network_or_llm():
    source = Path(apm.__file__).read_text(encoding="utf-8")
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
        "build_memory_proposals", "agent.workbench_service", "agent.memory_ledger",
        "agent.source_registry", "agent.pack_builder", "agent.project_packs",
        "agent.hf_knowledge_pack_importer", "agent.pdf_pack_importer",
        "datasets", "huggingface_hub", "requests", "urllib", "urllib.request",
        "httpx", "openai", "anthropic",
    }
    leaked = imported & forbidden
    assert not leaked, f"monitor must not import forbidden deps: {leaked}"

    body = source.replace(ast.get_docstring(tree, clean=False) or "", "", 1)
    for token in ("ActivationStateManager(", "MemoryLedger(", "save_registry(",
                  "propose_source", "load_dataset", "hf_hub_download",
                  "requests.get", "urlopen", "OpenAI(", "Anthropic(",
                  ".activate(", ".deactivate(", ".rollback(", ".supersede("):
        assert token not in body, f"monitor must not reference {token!r}"


def test_module_only_imports_stdlib_and_readonly_activation():
    source = Path(apm.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    roots = set()
    activation_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
            if node.module == "agent.knowledge_pack_activation":
                activation_names.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
    stdlib_and_agent = {
        "__future__", "hashlib", "json", "os", "tempfile", "dataclasses",
        "datetime", "enum", "pathlib", "typing", "agent",
    }
    assert roots <= stdlib_and_agent, f"unexpected imports: {roots - stdlib_and_agent}"
    # Only read-only activation names are imported (no writer/manager).
    assert "ActivationStateManager" not in activation_names
    assert activation_names <= {
        "ACTIVATION_LAYER_VERSION", "ActivationStateManifest", "ActivePackState",
        "CoexistencePolicy", "PackLifecycleState", "select_active_pack_ids",
    }
