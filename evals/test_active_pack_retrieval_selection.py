"""v7.0 retrieval selection adapter + module import purity.

Pins that the active-pack selection adapter exposes only active, environment-
matched packs (ignoring imported/inactive/superseded/retired/blocked), resolves
them to on-disk directories, and that the activation module imports no memory,
registry, proposal, pack-content-writer, retrieval, or LLM dependency and touches
no network.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import knowledge_pack_activation as kpa  # noqa: E402
from _kp_activation_helpers import (  # noqa: E402
    FIXED_NOW,
    approval_for,
    passing_evidence,
    write_hf_pack,
    write_pdf_pack,
)


def _manager(tmp_path):
    return kpa.ActivationStateManager(
        state_path=tmp_path / "config" / "state.jsonl",
        audit_path=tmp_path / "reports" / "audit.jsonl")


def _activate(mgr, identity, *, environment="default"):
    evidence = passing_evidence(identity)
    request = kpa.KnowledgePackActivationRequest(
        identity=identity, approval=approval_for(
            identity, evidence=evidence, environment=environment),
        evidence=evidence, environment=environment,
        current_state=kpa.PackLifecycleState.EVALUATED)
    return mgr.activate(request, write=True, now=FIXED_NOW)


def test_selection_only_returns_active(tmp_path):
    packs = tmp_path / "packs"
    hf = kpa.load_pack_identity(write_hf_pack(packs / "hf-demo"))
    pdf = kpa.load_pack_identity(write_pdf_pack(packs / "pdf-demo"))
    mgr = _manager(tmp_path)
    _activate(mgr, hf)
    _activate(mgr, pdf)
    mgr.deactivate("pdf-demo", emergency=True, actor="op", reason="pause",
                   write=True, now=FIXED_NOW)
    state = mgr.load_state()
    assert kpa.select_active_pack_ids(state) == ["hf-demo"]


def test_selection_resolves_dirs(tmp_path):
    packs = tmp_path / "packs"
    hf = kpa.load_pack_identity(write_hf_pack(packs / "hf-demo"))
    mgr = _manager(tmp_path)
    _activate(mgr, hf)
    dirs = kpa.select_active_pack_dirs(mgr.load_state(), packs)
    assert [d.name for d in dirs] == ["hf-demo"]
    assert (dirs[0] / "manifest.json").exists()


def test_empty_state_selects_nothing(tmp_path):
    assert kpa.select_active_pack_ids(kpa.ActivationStateManifest()) == []
    assert kpa.select_active_pack_dirs(
        kpa.ActivationStateManifest(), tmp_path) == []


def test_environment_scoping(tmp_path):
    packs = tmp_path / "packs"
    hf = kpa.load_pack_identity(write_hf_pack(packs / "hf-demo"))
    mgr = _manager(tmp_path)
    _activate(mgr, hf, environment="staging")
    state = mgr.load_state()
    assert kpa.select_active_pack_ids(state, environment="staging") == ["hf-demo"]
    assert kpa.select_active_pack_ids(state, environment="production") == []


def test_superseded_pack_not_selected(tmp_path):
    packs = tmp_path / "packs"
    v1 = kpa.load_pack_identity(write_hf_pack(
        packs / "hf-v1", pack_id="hf-v1", dataset_revision="rev-1"))
    v2 = kpa.load_pack_identity(write_hf_pack(
        packs / "hf-v2", pack_id="hf-v2", dataset_revision="rev-2"))
    mgr = _manager(tmp_path)
    _activate(mgr, v1)
    evidence2 = passing_evidence(v2)
    request = kpa.KnowledgePackActivationRequest(
        identity=v2, approval=approval_for(
            v2, scope=kpa.ActivationScope.SUPERSEDE, evidence=evidence2),
        evidence=evidence2)
    mgr.supersede(old_pack_id="hf-v1", request=request, write=True, now=FIXED_NOW)
    assert kpa.select_active_pack_ids(mgr.load_state()) == ["hf-v2"]


def test_module_imports_no_forbidden_dependencies():
    source = Path(kpa.__file__).read_text(encoding="utf-8")
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
        "MemoryLedger", "save_registry", "save_memory_review_queue",
        "save_source_review_queue", "propose_source_updates",
        "build_memory_proposals",
        "agent.memory_ledger", "agent.source_registry", "agent.workbench_service",
        "agent.retrieval", "agent.hybrid_retrieval", "agent.pack_builder",
        "agent.project_packs", "agent.knowledge_packs", "agent.knowledge_library",
        "agent.hf_knowledge_pack_importer", "agent.pdf_pack_importer",
        "WorkbenchService", "PackRegistry", "KnowledgeLibrary",
        "datasets", "huggingface_hub", "requests", "urllib", "urllib.request",
        "httpx", "openai", "anthropic", "pytesseract", "pdf2image",
    }
    leaked = imported & forbidden
    assert not leaked, f"module must not import forbidden deps: {leaked}"

    body = source.replace(ast.get_docstring(tree, clean=False) or "", "", 1)
    for token in (
        "load_dataset", "hf_hub_download", "snapshot_download", "requests.get",
        "urlopen", "MemoryLedger(", "save_registry(", "propose_source",
        "OpenAI(", "Anthropic(",
    ):
        assert token not in body, f"module must not reference {token!r}"


def test_module_only_imports_stdlib():
    source = Path(kpa.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
    stdlib = {
        "__future__", "hashlib", "json", "os", "tempfile", "dataclasses",
        "datetime", "enum", "pathlib", "typing",
    }
    assert roots <= stdlib, f"unexpected non-stdlib imports: {roots - stdlib}"
