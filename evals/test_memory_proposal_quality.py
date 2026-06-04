"""Tests for the v5.0 memory proposal quality upgrade (offline, no downloads).

These prove the *proposal-quality* contract:

* candidates are typed (FACT / DECISION / PROJECT_STATE / USER_PREFERENCE /
  TODO / OPEN_QUESTION / SOURCE_GAP / INVALID_CANDIDATE);
* factual memory types are evidence-bound; non-approvable junk (vague,
  operator-task, unsupported) is surfaced as INVALID_CANDIDATE *with a reason*,
  never silently dropped;
* proposal ids, rendering, and export are deterministic;
* nothing here writes to the MemoryLedger and the module imports no ledger/bank
  write path at all;
* the retrieval path is unchanged when proposals are built alongside it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import memory_proposal_quality as mpq  # noqa: E402
from agent.memory_proposal_quality import (  # noqa: E402
    MemoryProposalType,
    RawMemoryCandidate,
)
from agent import memory_ledger as memory_ledger_module  # noqa: E402
from agent.memory_ledger import MemoryLedger  # noqa: E402
from agent.project_packs import PackRegistry  # noqa: E402
from agent.workbench_service import WorkbenchService  # noqa: E402
from agent import pack_builder  # noqa: E402
from agent import retrieval_eval_harness as reh  # noqa: E402
from retrieval.embedding_backend import OfflineHashingEmbedder  # noqa: E402

_DEMO = ROOT / "demos" / "memory_proposal_candidates.jsonl"
_MANIFEST = ROOT / "packs" / "m365_coding_assistant" / "pack.yaml"
_CASES = ROOT / "demos" / "retrieval_eval_cases.jsonl"


def _supported_fact() -> RawMemoryCandidate:
    return RawMemoryCandidate(
        text="Pydantic raises ValidationError on invalid input at construction.",
        evidence_text="Pydantic validates fields and raises ValidationError on "
                      "invalid input.",
        evidence_source_id="pydantic-docs",
        evidence_chunk_id="pydantic-docs#3",
        suggested_type="fact",
    )


# 1. factual claims are FACT only when evidence-supported. -------------------

def test_supported_claim_is_fact():
    proposal = mpq.build_proposal(_supported_fact())
    assert proposal.proposal_type == MemoryProposalType.FACT
    assert proposal.invalid_reason == ""
    assert mpq.is_approvable_as_memory(proposal)


def test_unsupported_factual_claim_is_not_fact():
    candidate = RawMemoryCandidate(
        text="The retrieval system is the fastest in the industry.")
    proposal = mpq.build_proposal(candidate)
    assert proposal.proposal_type == MemoryProposalType.INVALID_CANDIDATE
    assert proposal.invalid_reason
    assert not mpq.is_approvable_as_memory(proposal)


# 2. decisions are classified separately from facts. ------------------------

def test_decision_classified_separately_from_fact():
    candidate = RawMemoryCandidate(
        text="The team decided to use the hybrid retrieval backend by default.",
        evidence_text="ADR: we chose the hybrid retrieval backend as default.",
        evidence_source_id="adr-retrieval",
        evidence_chunk_id="adr-retrieval#1",
    )
    proposal = mpq.build_proposal(candidate)
    assert proposal.proposal_type == MemoryProposalType.DECISION
    assert proposal.proposal_type != MemoryProposalType.FACT
    assert mpq.is_factual_memory_type(proposal.proposal_type)


def test_decision_without_evidence_is_invalid():
    candidate = RawMemoryCandidate(
        text="We decided to adopt the hybrid backend.")
    proposal = mpq.build_proposal(candidate)
    assert proposal.proposal_type == MemoryProposalType.INVALID_CANDIDATE
    assert "evidence" in proposal.invalid_reason.lower()


# 3. project-state memories are classified correctly. -----------------------

def test_project_state_classified_correctly():
    candidate = RawMemoryCandidate(
        text="The source registry currently exposes four layers.",
        evidence_text="The registry now exposes registry, audit, proposal, and "
                      "review-queue layers.",
        evidence_source_id="build-notes",
        evidence_chunk_id="build-notes#7",
    )
    proposal = mpq.build_proposal(candidate)
    assert proposal.proposal_type == MemoryProposalType.PROJECT_STATE
    assert proposal.staleness_risk == "high"


# 4. TODO / open-question / source-gap are not factual memory. --------------

def test_todo_open_question_source_gap_are_not_factual():
    todo = mpq.build_proposal(RawMemoryCandidate(
        text="TODO: migrate Redis to Managed Redis before the Q3 cutover."))
    question = mpq.build_proposal(RawMemoryCandidate(
        text="Is deterministic paraphrase retrieval acceptable for reports?"))
    gap = mpq.build_proposal(RawMemoryCandidate(
        text="No source documents the data-residency guarantees."))

    assert todo.proposal_type == MemoryProposalType.TODO
    assert question.proposal_type == MemoryProposalType.OPEN_QUESTION
    assert gap.proposal_type == MemoryProposalType.SOURCE_GAP
    for proposal in (todo, question, gap):
        assert not mpq.is_factual_memory_type(proposal.proposal_type)
        assert mpq.is_tracked_non_factual_type(proposal.proposal_type)
        assert not mpq.is_approvable_as_memory(proposal)
        assert proposal.invalid_reason == ""


# 5. vague placeholder text becomes INVALID_CANDIDATE. ----------------------

def test_vague_placeholder_is_invalid():
    for text in ("TBD", "   ", "stuff etc", "fixme: ..."):
        proposal = mpq.build_proposal(RawMemoryCandidate(text=text))
        assert proposal.proposal_type == MemoryProposalType.INVALID_CANDIDATE
        assert proposal.invalid_reason
        assert not mpq.is_approvable_as_memory(proposal)


# 6. operator instructions become INVALID_CANDIDATE with a reason. ----------

def test_operator_instruction_is_invalid_with_reason():
    candidate = RawMemoryCandidate(
        text="Commit and push v4.3 and then run pytest.", suggested_type="fact")
    proposal = mpq.build_proposal(candidate)
    assert proposal.proposal_type == MemoryProposalType.INVALID_CANDIDATE
    assert "instruction" in proposal.invalid_reason.lower()
    assert not mpq.is_approvable_as_memory(proposal)


# 7. user preference is approvable without a document source. ----------------

def test_user_preference_is_approvable_without_evidence():
    candidate = RawMemoryCandidate(
        text="The user prefers additive, approval-gated changes.")
    proposal = mpq.build_proposal(candidate)
    assert proposal.proposal_type == MemoryProposalType.USER_PREFERENCE
    assert mpq.is_factual_memory_type(proposal.proposal_type)
    assert mpq.is_approvable_as_memory(proposal)


# 8. every proposal requires approval and stays "proposed". -----------------

def test_every_proposal_requires_approval_and_is_proposed():
    proposals = mpq.build_memory_proposals(
        mpq.load_memory_candidates(_DEMO))
    assert proposals
    for proposal in proposals:
        assert proposal.requires_human_approval is True
        assert proposal.status == "proposed"


# 9. proposal ids are deterministic and type-stable. ------------------------

def test_proposal_ids_are_deterministic():
    candidate = _supported_fact()
    first = mpq.build_proposal(candidate)
    second = mpq.build_proposal(candidate)
    assert first.proposal_id == second.proposal_id
    assert first.proposal_id.startswith("memprop-")


def test_proposal_id_is_stable_across_reclassification():
    # Same claim + evidence binding, different upstream hint -> same id.
    base = _supported_fact()
    rehinted = RawMemoryCandidate(
        text=base.text, evidence_text=base.evidence_text,
        evidence_source_id=base.evidence_source_id,
        evidence_chunk_id=base.evidence_chunk_id, suggested_type="decision")
    assert (mpq.build_proposal(base).proposal_id
            == mpq.build_proposal(rehinted).proposal_id)


# 10. rendering and export are deterministic. -------------------------------

def test_render_and_export_are_deterministic():
    candidates = mpq.load_memory_candidates(_DEMO)
    proposals_a = mpq.build_memory_proposals(candidates)
    proposals_b = mpq.build_memory_proposals(candidates)
    assert (mpq.render_memory_proposals_markdown(proposals_a)
            == mpq.render_memory_proposals_markdown(proposals_b))
    assert (mpq.memory_proposals_to_jsonl(proposals_a)
            == mpq.memory_proposals_to_jsonl(proposals_b))


def test_write_memory_proposals_roundtrips(tmp_path):
    proposals = mpq.build_memory_proposals(mpq.load_memory_candidates(_DEMO))
    out = tmp_path / "export.jsonl"
    mpq.write_memory_proposals(proposals, out)
    assert out.exists()
    reloaded = [
        mpq.MemoryProposalQuality.from_dict(json.loads(line))
        for line in out.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [p.proposal_id for p in reloaded] == [p.proposal_id for p in proposals]


# 11. demo dataset yields the expected typed baseline. ----------------------

def test_demo_dataset_baseline_counts():
    proposals = mpq.build_memory_proposals(mpq.load_memory_candidates(_DEMO))
    by_type: dict = {}
    for proposal in proposals:
        by_type[proposal.proposal_type] = by_type.get(
            proposal.proposal_type, 0) + 1
    assert by_type.get(MemoryProposalType.FACT) == 1
    assert by_type.get(MemoryProposalType.DECISION) == 1
    assert by_type.get(MemoryProposalType.PROJECT_STATE) == 1
    assert by_type.get(MemoryProposalType.USER_PREFERENCE) == 1
    assert by_type.get(MemoryProposalType.TODO) == 1
    assert by_type.get(MemoryProposalType.OPEN_QUESTION) == 1
    assert by_type.get(MemoryProposalType.SOURCE_GAP) == 1
    assert by_type.get(MemoryProposalType.INVALID_CANDIDATE) == 3
    approvable = [p for p in proposals if mpq.is_approvable_as_memory(p)]
    assert len(approvable) == 4  # FACT, DECISION, PROJECT_STATE, USER_PREFERENCE


# 12. no MemoryLedger write occurs (spy + module-import purity). -------------

def test_building_proposals_does_not_write_memory_ledger(tmp_path, monkeypatch):
    calls = []
    original_add = MemoryLedger.add

    def _spy(self, *args, **kwargs):
        calls.append(args)
        return original_add(self, *args, **kwargs)

    monkeypatch.setattr(MemoryLedger, "add", _spy)

    proposals = mpq.build_memory_proposals(mpq.load_memory_candidates(_DEMO))
    mpq.write_memory_proposals(proposals, tmp_path / "export.jsonl")
    mpq.render_memory_proposals_markdown(proposals)

    assert calls == []


def test_module_has_no_ledger_or_bank_write_imports():
    # Inspect import statements only (the docstring may *mention* these names
    # to explain what the module deliberately avoids).
    source = Path(mpq.__file__).read_text(encoding="utf-8")
    import_lines = [
        line.strip() for line in source.splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    banned = ("memory_ledger", "memory_bank", "MemoryLedger", "MemoryBank",
              "add_memory")
    for line in import_lines:
        assert not any(token in line for token in banned), line
    # The runtime namespace must not expose any ledger/bank write symbol.
    for symbol in ("MemoryLedger", "MemoryBank", "add_memory"):
        assert not hasattr(mpq, symbol)


# 13. retrieval path is unchanged when proposals are built alongside it. -----

def _build_service(tmp_path, monkeypatch) -> WorkbenchService:
    monkeypatch.chdir(ROOT)
    registry = PackRegistry(tmp_path / "packs")
    plan = pack_builder.PackBuildPlan.from_file(_MANIFEST)
    report = pack_builder.build_pack(plan, registry)
    pack = registry.get_pack(report.pack_id)
    return WorkbenchService.from_pack(
        pack, registry=registry, knowledge_backend="hybrid",
        semantic_embedder=OfflineHashingEmbedder())


def test_retrieval_unchanged_when_building_proposals(tmp_path, monkeypatch):
    service = _build_service(tmp_path, monkeypatch)
    cases = reh.load_cases(_CASES)

    before = reh.summarize(reh.run_eval(service, cases)).to_dict()
    # Build memory proposals in between two identical retrieval evaluations.
    mpq.build_memory_proposals(mpq.load_memory_candidates(_DEMO))
    after = reh.summarize(reh.run_eval(service, cases)).to_dict()

    assert before == after
