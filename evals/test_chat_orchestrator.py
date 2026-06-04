"""Tests for the v6.0 governed chat orchestrator (read-only).

The orchestrator routes a user question into a *safe* answer mode and renders a
result. It may answer, cite, label judgement, refuse, or emit follow-up
*proposals* for human review — and it must never mutate durable state. These
tests pin that contract:

* the deterministic router classifies evidence, judgement, memory-write, and
  source-maintenance queries into the right first-pass modes;
* an evidence-backed query is answered with its citations preserved;
* a query with no citable evidence is refused as INSUFFICIENT_EVIDENCE naming
  the gap;
* a judgement query always labels its judgement; no unlabelled judgement leaks;
* a memory-write instruction yields a *proposal only* (no MemoryLedger write);
* a source-maintenance request yields *proposals only* (no registry write);
* ``state_mutation_attempted`` is always ``False``;
* the v3.0 retrieval baseline is unchanged when chat runs alongside it;
* the source registry and memory review queue files stay byte-identical;
* ``MemoryLedger.add`` is never called;
* CLI output is deterministic.

Chat may answer; chat must not mutate durable state.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import chat_orchestrator as co  # noqa: E402
from agent.chat_orchestrator import (  # noqa: E402
    ChatMode,
    ChatOrchestrator,
    route_query,
)
from agent import memory_proposal_quality as mpq  # noqa: E402
from agent.memory_proposal_quality import MemoryReviewStatus  # noqa: E402
from agent import pack_builder  # noqa: E402
from agent import retrieval_eval_harness as reh  # noqa: E402
from agent.memory_ledger import MemoryLedger  # noqa: E402
from agent.project_packs import PackRegistry  # noqa: E402
from agent.workbench_service import WorkbenchService  # noqa: E402
from slm.assistant_composer import JUDGEMENT_LABEL  # noqa: E402
from retrieval.embedding_backend import OfflineHashingEmbedder  # noqa: E402

_MANIFEST = ROOT / "packs" / "m365_coding_assistant" / "pack.yaml"
_CASES = ROOT / "demos" / "retrieval_eval_cases.jsonl"
_REGISTRY = ROOT / "demos" / "source_registry.jsonl"
_CANDIDATES = ROOT / "demos" / "memory_proposal_candidates.jsonl"
_README = ROOT / "README.md"

_NOW = datetime(2026, 6, 4, tzinfo=timezone.utc)

# An evidence-backed query: least-privilege/PIM grounds in the admin-patterns
# source under the hybrid backend (see demos/retrieval_eval_cases.jsonl).
_EVIDENCE_QUERY = (
    "How do we set up least-privilege admin roles, separate break-glass "
    "Global Administrator accounts, and Privileged Identity Management for "
    "just-in-time activation?")


def _build_service(tmp_path, monkeypatch) -> WorkbenchService:
    monkeypatch.chdir(ROOT)
    registry = PackRegistry(tmp_path / "packs")
    plan = pack_builder.PackBuildPlan.from_file(_MANIFEST)
    report = pack_builder.build_pack(plan, registry)
    pack = registry.get_pack(report.pack_id)
    # seed=False equivalent: no memory is added, so answers are knowledge-only.
    return WorkbenchService.from_pack(
        pack, registry=registry, knowledge_backend="hybrid",
        semantic_embedder=OfflineHashingEmbedder())


def _orchestrator(tmp_path, monkeypatch) -> ChatOrchestrator:
    service = _build_service(tmp_path, monkeypatch)
    return ChatOrchestrator(service, registry_path=str(_REGISTRY))


# 1. evidence query routes to EVIDENCE_ANSWER. -------------------------------

def test_router_evidence_query_routes_to_evidence_answer():
    intent = route_query("What is least-privilege access control?")
    assert intent.mode == ChatMode.EVIDENCE_ANSWER


# 2. a query with no citable evidence is refused as INSUFFICIENT_EVIDENCE. ----

def test_unsupported_query_routes_to_insufficient_evidence(tmp_path, monkeypatch):
    orch = _orchestrator(tmp_path, monkeypatch)
    # A factual question the pack has no source for (documented gap probe).
    result = orch.answer(
        "What is our decision on data residency and which Azure region stores "
        "customer data?")
    assert result.mode == ChatMode.INSUFFICIENT_EVIDENCE
    assert result.citations == []
    assert result.refusal_reason  # names the evidence gap


# 3. a judgement query labels its judgement. ---------------------------------

def test_judgement_query_labels_judgement(tmp_path, monkeypatch):
    orch = _orchestrator(tmp_path, monkeypatch)
    result = orch.answer(
        "Should we switch the default retrieval backend to deterministic?")
    assert result.mode == ChatMode.JUDGEMENT_ONLY
    assert result.judgement_labelled is True
    assert JUDGEMENT_LABEL in result.answer_text


# 4. a memory-write instruction yields a proposal only, not a ledger write. ---

def test_memory_instruction_generates_proposal_not_write(tmp_path, monkeypatch):
    orch = _orchestrator(tmp_path, monkeypatch)
    result = orch.answer(
        "Remember that the team chose hybrid retrieval as the default backend.")
    assert result.mode == ChatMode.PROPOSE_MEMORY
    assert result.proposed_memory_count >= 1
    # Every emitted proposal requires human approval and is only "proposed".
    for proposal in result.proposed_memory:
        assert proposal["requires_human_approval"] is True
        assert proposal["status"] == "proposed"
    assert result.state_mutation_attempted is False


# 5. a source-maintenance query yields proposals only, not a registry write. --

def test_source_query_generates_proposal_not_write(tmp_path, monkeypatch):
    orch = _orchestrator(tmp_path, monkeypatch)
    before = _REGISTRY.read_bytes()
    result = orch.answer(
        "Run a source registry audit and propose source updates for any stale "
        "or owner-less sources.")
    assert result.mode == ChatMode.PROPOSE_SOURCE_UPDATE
    assert result.proposed_source_update_count >= 1
    for proposal in result.proposed_source_updates:
        assert proposal["requires_human_approval"] is True
        assert proposal["status"] == "proposed"
    # The registry file is untouched.
    assert _REGISTRY.read_bytes() == before


# 6. state_mutation_attempted remains false across every mode. ----------------

def test_state_mutation_attempted_always_false(tmp_path, monkeypatch):
    orch = _orchestrator(tmp_path, monkeypatch)
    queries = [
        _EVIDENCE_QUERY,
        "What is our decision on data residency?",
        "Should we adopt hybrid retrieval everywhere?",
        "Remember that hybrid is the default backend.",
        "Audit the source registry and propose source updates.",
        "Delete the source registry file.",
    ]
    for query in queries:
        result = orch.answer(query)
        assert result.state_mutation_attempted is False


# 7. the v3.0 retrieval baseline is unchanged when chat runs alongside it. ----

def test_retrieval_baseline_unchanged_with_chat(tmp_path, monkeypatch):
    service = _build_service(tmp_path, monkeypatch)
    cases = reh.load_cases(_CASES)
    before = [r.passed for r in reh.run_eval(service, cases)]

    orch = ChatOrchestrator(service, registry_path=str(_REGISTRY))
    orch.answer(_EVIDENCE_QUERY)
    orch.answer("Remember that hybrid is the default backend.")
    orch.answer("Audit the source registry and propose source updates.")

    after = [r.passed for r in reh.run_eval(service, cases)]
    assert before == after


# 8. the source registry file stays byte-identical. --------------------------

def test_source_registry_byte_identical(tmp_path, monkeypatch):
    orch = _orchestrator(tmp_path, monkeypatch)
    before = _REGISTRY.read_bytes()
    orch.answer("Audit the source registry and propose source updates.")
    orch.answer("Are there any stale source entries that need a new owner?")
    assert _REGISTRY.read_bytes() == before


# 9. a memory review queue stays byte-identical. -----------------------------

def test_memory_review_queue_byte_identical(tmp_path, monkeypatch):
    # Build a v5.1 review queue and snapshot it.
    proposals = mpq.build_memory_proposals(
        mpq.load_memory_candidates(_CANDIDATES))
    queue = mpq.import_memory_proposals_to_queue(
        [p.to_dict() for p in proposals], [])
    queue = mpq.apply_memory_review_to_queue(
        queue, queue[0].proposal_id, MemoryReviewStatus.APPROVED)
    queue_path = tmp_path / "queue.jsonl"
    mpq.save_memory_review_queue(queue, queue_path)
    before = queue_path.read_bytes()

    # Running chat (incl. a memory proposal) does not touch the queue file.
    orch = _orchestrator(tmp_path, monkeypatch)
    orch.answer("Remember that hybrid retrieval is the default backend.")
    assert queue_path.read_bytes() == before


# 10. MemoryLedger.add is never called. --------------------------------------

def test_memory_ledger_add_never_called(tmp_path, monkeypatch):
    calls = []
    original_add = MemoryLedger.add

    def _spy(self, *args, **kwargs):
        calls.append(args)
        return original_add(self, *args, **kwargs)

    monkeypatch.setattr(MemoryLedger, "add", _spy)

    orch = _orchestrator(tmp_path, monkeypatch)
    orch.answer(_EVIDENCE_QUERY)
    orch.answer("Remember that hybrid is the default backend.")
    orch.answer("Audit the source registry and propose source updates.")
    orch.answer("Should we switch backends?")

    assert calls == []


# 11. answer citations are preserved for an evidence-backed response. ---------

def test_citations_preserved_for_evidence_answer(tmp_path, monkeypatch):
    orch = _orchestrator(tmp_path, monkeypatch)
    result = orch.answer(_EVIDENCE_QUERY)
    assert result.mode == ChatMode.EVIDENCE_ANSWER
    assert result.citations  # grounded answer keeps its citations
    # The citations the orchestrator reports are exactly the composer's.
    direct = orch._service.answer_query(_EVIDENCE_QUERY)
    assert result.citations == list(direct.answer.citations)


# 12. no unlabelled judgement appears in any answer. -------------------------

def test_no_unlabelled_judgement(tmp_path, monkeypatch):
    orch = _orchestrator(tmp_path, monkeypatch)
    for query in [
        _EVIDENCE_QUERY,
        "Should we switch the default retrieval backend?",
        "What do you think is the best approach to least-privilege roles?",
    ]:
        result = orch.answer(query)
        # If the answer contains judgement prose, it must be labelled.
        if "interpretation" in result.answer_text.lower():
            assert JUDGEMENT_LABEL in result.answer_text
            assert result.judgement_labelled is True


# 13. CLI output is deterministic. -------------------------------------------

def test_cli_output_deterministic(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "app"))
    import workbench

    capsys.readouterr()
    assert workbench.main(["chat", "ask", _EVIDENCE_QUERY]) == 0
    out_a = capsys.readouterr().out
    assert "# Chat answer (v6.0; read-only)" in out_a
    assert "State mutation attempted: false" in out_a

    assert workbench.main(["chat", "ask", _EVIDENCE_QUERY]) == 0
    out_b = capsys.readouterr().out
    assert out_b == out_a


# 14. the orchestrator module imports no durable-state writer. ---------------

def test_orchestrator_imports_no_writer():
    import ast

    source = Path(co.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported.add(alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
    forbidden = {
        "MemoryLedger",
        "save_registry",
        "save_memory_review_queue",
        "write_memory_proposals",
        "write_proposals",
        "add_memory",
    }
    leaked = imported & forbidden
    assert not leaked, f"orchestrator must not import writers: {leaked}"
    # No writer is invoked anywhere in the module *code* (docstring excluded).
    code = source.replace(ast.get_docstring(tree, clean=False) or "", "", 1)
    assert "MemoryLedger" not in code
    assert ".add(" not in code
    assert "save_" not in code


# Router coverage for the remaining declared modes. --------------------------

def test_router_covers_remaining_modes():
    assert route_query("Remember that X is true.").mode == ChatMode.PROPOSE_MEMORY
    assert route_query(
        "Audit the source registry.").mode == ChatMode.PROPOSE_SOURCE_UPDATE
    assert route_query(
        "Delete the registry file.").mode == ChatMode.UNSUPPORTED_REQUEST
    assert route_query(
        "Write a consultant report on least-privilege roles.").mode == (
            ChatMode.REPORT_STYLE_ANSWER)
    assert route_query(
        "What do we know about our break-glass accounts?").mode == (
            ChatMode.MEMORY_CONTEXT)


# README documents the chat modes and the non-mutation guarantee. ------------

def test_readme_documents_chat_orchestrator():
    text = _README.read_text(encoding="utf-8")
    assert "## v6.0" in text
    assert "chat" in text.lower()
    assert "read-only" in text.lower()
