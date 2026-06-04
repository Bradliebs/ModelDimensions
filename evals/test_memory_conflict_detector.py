"""Tests for the v5.2 Memory Conflict + Staleness Detector.

The detector is **read-only**: it checks v5.0/v5.1 memory proposals against
existing memory and reports risk (duplicate, contradiction, supersession,
staleness, missing evidence, low confidence, not-writeable). These tests pin
that contract:

* each risk code is detected deterministically and conservatively;
* INVALID_CANDIDATE / TODO / OPEN_QUESTION / SOURCE_GAP are flagged as not
  directly writeable memory;
* detector output is deterministic;
* the CLI prints to stdout without writing files, and ``--out`` writes *only*
  the report;
* **nothing is written or mutated** — no ``MemoryLedger`` entry, no proposal
  queue change, no existing-memory change;
* the v5.0 proposal build, v5.1 review queue, and v3.0 retrieval baseline are
  unchanged when detection runs alongside them;
* the README documents the detector's purpose and limitations.

Conflict detection reports risk; it does not resolve it or write memory.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import memory_conflict_detector as mcd  # noqa: E402
from agent.memory_conflict_detector import (  # noqa: E402
    ExistingMemoryRecord,
    MemoryConflictCode,
    MemoryConflictSeverity,
)
from agent import memory_proposal_quality as mpq  # noqa: E402
from agent.memory_proposal_quality import (  # noqa: E402
    MemoryProposalQuality,
    MemoryProposalType,
    MemoryReviewStatus,
)
from agent import pack_builder  # noqa: E402
from agent import retrieval_eval_harness as reh  # noqa: E402
from agent.project_packs import PackRegistry  # noqa: E402
from agent.workbench_service import WorkbenchService  # noqa: E402
from agent.memory_ledger import MemoryLedger  # noqa: E402
from retrieval.embedding_backend import OfflineHashingEmbedder  # noqa: E402

_DEMO_CANDIDATES = ROOT / "demos" / "memory_proposal_candidates.jsonl"
_DEMO_EXISTING = ROOT / "demos" / "existing_memory_records.jsonl"
_MANIFEST = ROOT / "packs" / "m365_coding_assistant" / "pack.yaml"
_CASES = ROOT / "demos" / "retrieval_eval_cases.jsonl"
_README = ROOT / "README.md"

_NOW = datetime(2026, 6, 4, tzinfo=timezone.utc)


def _proposal(claim: str, proposal_type: str, *, proposal_id: str = "",
              confidence: float = 0.9, evidence_source_id: str = "src-1",
              evidence_text: str = "supporting evidence text",
              created_at=None, invalid_reason: str = "") -> MemoryProposalQuality:
    return MemoryProposalQuality(
        proposal_id=proposal_id or ("memprop-" + str(abs(hash(claim)) % 10**8)),
        proposal_type=proposal_type,
        claim=claim,
        evidence_text=evidence_text,
        evidence_source_id=evidence_source_id,
        confidence=confidence,
        created_at=created_at,
        invalid_reason=invalid_reason,
    )


def _codes(report) -> set:
    return {f.code for f in report.findings}


def _snapshot_tree(root: Path) -> set:
    return {p for p in root.rglob("*") if p.is_file()}


def _build_service(tmp_path, monkeypatch) -> WorkbenchService:
    monkeypatch.chdir(ROOT)
    registry = PackRegistry(tmp_path / "packs")
    plan = pack_builder.PackBuildPlan.from_file(_MANIFEST)
    report = pack_builder.build_pack(plan, registry)
    pack = registry.get_pack(report.pack_id)
    return WorkbenchService.from_pack(
        pack, registry=registry, knowledge_backend="hybrid",
        semantic_embedder=OfflineHashingEmbedder())


# 1. exact duplicate claim is detected. --------------------------------------

def test_exact_duplicate_claim_detected():
    claim = "Pydantic raises ValidationError on invalid input at construction."
    proposal = _proposal(claim, MemoryProposalType.FACT)
    existing = [ExistingMemoryRecord(
        memory_id="mem-1", memory_type="fact", claim=claim.upper())]
    report = mcd.detect_memory_conflicts([proposal], existing, now=_NOW)
    dup = [f for f in report.findings
           if f.code == MemoryConflictCode.DUPLICATE_CLAIM]
    assert len(dup) == 1
    assert dup[0].existing_id == "mem-1"
    assert dup[0].severity == MemoryConflictSeverity.WARNING


# 2. near duplicate claim is detected conservatively. ------------------------

def test_near_duplicate_detected_conservatively():
    existing_claim = ("The retrieval harness measures off-topic inclusion rate "
                      "and fail count.")
    proposal_claim = ("The retrieval harness measures off-topic inclusion rate "
                      "and fail count precisely.")
    proposal = _proposal(proposal_claim, MemoryProposalType.FACT)
    existing = [ExistingMemoryRecord(
        memory_id="mem-2", memory_type="fact", claim=existing_claim)]
    report = mcd.detect_memory_conflicts([proposal], existing, now=_NOW)
    assert MemoryConflictCode.POSSIBLE_DUPLICATE in _codes(report)
    # Conservative: a possible-duplicate is info, not an asserted duplicate.
    poss = [f for f in report.findings
            if f.code == MemoryConflictCode.POSSIBLE_DUPLICATE][0]
    assert poss.severity == MemoryConflictSeverity.INFO


# 3. contradiction is detected for simple deterministic cases. ---------------

def test_contradiction_detected_on_polarity_flip():
    proposal = _proposal(
        "The default retrieval backend is hybrid for value sprints.",
        MemoryProposalType.DECISION)
    existing = [ExistingMemoryRecord(
        memory_id="mem-3", memory_type="decision",
        claim="The default retrieval backend is not hybrid for value sprints.")]
    report = mcd.detect_memory_conflicts([proposal], existing, now=_NOW)
    contra = [f for f in report.findings
              if f.code == MemoryConflictCode.CONTRADICTS_EXISTING_MEMORY]
    assert len(contra) == 1
    assert contra[0].severity == MemoryConflictSeverity.ERROR
    assert contra[0].existing_id == "mem-3"


# 4. supersession is detected for project-state updates. ---------------------

def test_supersession_detected_for_project_state():
    proposal = _proposal(
        "The source registry currently exposes registry, audit, and review "
        "layers.", MemoryProposalType.PROJECT_STATE,
        created_at="2026-06-01T00:00:00+00:00")
    older = ExistingMemoryRecord(
        memory_id="mem-old", memory_type="project_state",
        claim="The source registry currently exposes registry and audit layers.",
        created_at="2026-01-01T00:00:00+00:00")
    report = mcd.detect_memory_conflicts([proposal], [older], now=_NOW)
    assert MemoryConflictCode.SUPERSEDES_EXISTING_MEMORY in _codes(report)

    # The reverse direction is reported as superseded_by when existing is newer.
    proposal_old = _proposal(
        "The source registry currently exposes registry and audit layers.",
        MemoryProposalType.PROJECT_STATE,
        created_at="2026-01-01T00:00:00+00:00")
    newer = ExistingMemoryRecord(
        memory_id="mem-new", memory_type="project_state",
        claim="The source registry currently exposes registry, audit, and "
              "review layers.",
        created_at="2026-06-01T00:00:00+00:00")
    rev = mcd.detect_memory_conflicts([proposal_old], [newer], now=_NOW)
    assert MemoryConflictCode.SUPERSEDED_BY_EXISTING_MEMORY in _codes(rev)


# 5. stale project state is flagged. -----------------------------------------

def test_stale_project_state_flagged():
    proposal = _proposal(
        "The build currently targets the v3 retrieval baseline.",
        MemoryProposalType.PROJECT_STATE,
        created_at="2025-01-01T00:00:00+00:00")
    report = mcd.detect_memory_conflicts([proposal], [], now=_NOW)
    stale = [f for f in report.findings
             if f.code == MemoryConflictCode.STALE_PROJECT_STATE]
    assert len(stale) == 1
    assert stale[0].severity == MemoryConflictSeverity.WARNING


# 6. stale user preference is flagged only conservatively. -------------------

def test_stale_user_preference_flagged_only_conservatively():
    old_pref = _proposal(
        "The user prefers additive, approval-gated changes.",
        MemoryProposalType.USER_PREFERENCE, confidence=0.7,
        created_at="2025-01-01T00:00:00+00:00")
    recent_pref = _proposal(
        "The user prefers concise commit messages.",
        MemoryProposalType.USER_PREFERENCE, confidence=0.7,
        created_at="2026-04-01T00:00:00+00:00")

    old_report = mcd.detect_memory_conflicts([old_pref], [], now=_NOW)
    recent_report = mcd.detect_memory_conflicts([recent_pref], [], now=_NOW)

    # > 365 days old: flagged, but only at info severity (conservative).
    stale = [f for f in old_report.findings
             if f.code == MemoryConflictCode.STALE_USER_PREFERENCE]
    assert len(stale) == 1
    assert stale[0].severity == MemoryConflictSeverity.INFO
    # A recent preference is not flagged as stale.
    assert MemoryConflictCode.STALE_USER_PREFERENCE not in _codes(recent_report)


# 7. missing evidence is flagged for evidence-required types. ----------------

def test_missing_evidence_flagged_for_evidence_required_type():
    proposal = _proposal(
        "The encoder uses a deterministic hashing embedder offline.",
        MemoryProposalType.FACT, evidence_source_id="", evidence_text="")
    report = mcd.detect_memory_conflicts([proposal], [], now=_NOW)
    assert MemoryConflictCode.MISSING_EVIDENCE in _codes(report)


# 8. low confidence is flagged. ----------------------------------------------

def test_low_confidence_flagged():
    proposal = _proposal(
        "The hybrid backend slightly improves recall.",
        MemoryProposalType.FACT, confidence=0.4)
    report = mcd.detect_memory_conflicts([proposal], [], now=_NOW)
    low = [f for f in report.findings
           if f.code == MemoryConflictCode.LOW_CONFIDENCE]
    assert len(low) == 1
    assert low[0].severity == MemoryConflictSeverity.WARNING


# 9. invalid candidates are not writeable. -----------------------------------

def test_invalid_candidate_not_writeable():
    proposal = _proposal(
        "TBD", MemoryProposalType.INVALID_CANDIDATE, confidence=0.0,
        evidence_source_id="", evidence_text="",
        invalid_reason="vague or placeholder text")
    report = mcd.detect_memory_conflicts([proposal], [], now=_NOW)
    codes = _codes(report)
    assert MemoryConflictCode.INVALID_CANDIDATE in codes
    inv = [f for f in report.findings
           if f.code == MemoryConflictCode.INVALID_CANDIDATE][0]
    assert inv.severity == MemoryConflictSeverity.ERROR
    # An invalid candidate is never marked as needing approval into memory.
    assert MemoryConflictCode.NEEDS_HUMAN_REVIEW not in codes


# 10. TODO / OPEN_QUESTION / SOURCE_GAP are not writeable. -------------------

def test_tracked_non_factual_types_not_writeable():
    for proposal_type in (MemoryProposalType.TODO,
                          MemoryProposalType.OPEN_QUESTION,
                          MemoryProposalType.SOURCE_GAP):
        proposal = _proposal(
            "tracked item of some kind", proposal_type, confidence=0.4,
            evidence_source_id="", evidence_text="")
        report = mcd.detect_memory_conflicts([proposal], [], now=_NOW)
        codes = _codes(report)
        assert MemoryConflictCode.NON_FACTUAL_NOT_WRITEABLE in codes
        assert MemoryConflictCode.NEEDS_HUMAN_REVIEW not in codes


# 11. detector output is deterministic. --------------------------------------

def test_detector_output_is_deterministic():
    proposals = mpq.build_memory_proposals(
        mpq.load_memory_candidates(_DEMO_CANDIDATES))
    existing = mcd.load_existing_memory(_DEMO_EXISTING)
    report_a = mcd.detect_memory_conflicts(proposals, existing, now=_NOW)
    report_b = mcd.detect_memory_conflicts(proposals, existing, now=_NOW)
    assert report_a.to_jsonl() == report_b.to_jsonl()
    assert mcd.render_conflict_report_markdown(report_a) == \
        mcd.render_conflict_report_markdown(report_b)
    md = mcd.render_conflict_report_markdown(report_a)
    assert "# Memory conflict + staleness report (v5.2)" in md
    assert "does not resolve it or write memory" in md


def _demo_proposals_export(tmp_path) -> Path:
    proposals = mpq.build_memory_proposals(
        mpq.load_memory_candidates(_DEMO_CANDIDATES))
    out = tmp_path / "proposals.jsonl"
    mpq.write_memory_proposals(proposals, out)
    return out


# 12. CLI stdout writes no files. --------------------------------------------

def test_cli_stdout_writes_no_files(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "app"))
    import workbench

    proposals_path = _demo_proposals_export(tmp_path)
    existing_path = tmp_path / "existing.jsonl"
    existing_path.write_bytes(_DEMO_EXISTING.read_bytes())

    before = _snapshot_tree(tmp_path)
    capsys.readouterr()
    assert workbench.main([
        "memory-proposals", "check-conflicts", str(proposals_path),
        "--existing-memory", str(existing_path), "--now", _NOW.isoformat()]) == 0
    out_a = capsys.readouterr().out
    assert _snapshot_tree(tmp_path) == before  # nothing written
    assert "# Memory conflict + staleness report (v5.2)" in out_a

    # Deterministic across runs.
    assert workbench.main([
        "memory-proposals", "check-conflicts", str(proposals_path),
        "--existing-memory", str(existing_path), "--now", _NOW.isoformat()]) == 0
    assert capsys.readouterr().out == out_a


# 13. CLI --out writes only the report file. ---------------------------------

def test_cli_out_writes_only_report(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "app"))
    import workbench

    proposals_path = _demo_proposals_export(tmp_path)
    existing_path = tmp_path / "existing.jsonl"
    existing_path.write_bytes(_DEMO_EXISTING.read_bytes())
    proposals_before = proposals_path.read_bytes()
    existing_before = existing_path.read_bytes()

    report_path = tmp_path / "report.jsonl"
    before = _snapshot_tree(tmp_path)
    assert workbench.main([
        "memory-proposals", "check-conflicts", str(proposals_path),
        "--existing-memory", str(existing_path), "--out", str(report_path),
        "--now", _NOW.isoformat()]) == 0

    # Only the report file is new; inputs are byte-identical.
    assert _snapshot_tree(tmp_path) - before == {report_path}
    assert proposals_path.read_bytes() == proposals_before
    assert existing_path.read_bytes() == existing_before
    assert report_path.read_text(encoding="utf-8").strip()


# 14. MemoryLedger.add is never called. --------------------------------------

def test_memory_ledger_add_never_called(tmp_path, monkeypatch):
    calls = []
    original_add = MemoryLedger.add

    def _spy(self, *args, **kwargs):
        calls.append(args)
        return original_add(self, *args, **kwargs)

    monkeypatch.setattr(MemoryLedger, "add", _spy)

    proposals = mpq.build_memory_proposals(
        mpq.load_memory_candidates(_DEMO_CANDIDATES))
    existing = mcd.load_existing_memory(_DEMO_EXISTING)
    report = mcd.detect_memory_conflicts(proposals, existing, now=_NOW)
    mcd.write_conflict_report(report, tmp_path / "report.jsonl")
    mcd.render_conflict_report_markdown(report)

    assert calls == []


# 15. proposal queue remains byte-identical. ---------------------------------

def test_proposal_queue_byte_identical_after_check(tmp_path):
    proposals = mpq.build_memory_proposals(
        mpq.load_memory_candidates(_DEMO_CANDIDATES))
    queue = mpq.import_memory_proposals_to_queue(
        [p.to_dict() for p in proposals], [])
    queue = mpq.apply_memory_review_to_queue(
        queue, queue[0].proposal_id, MemoryReviewStatus.APPROVED)
    queue_path = tmp_path / "queue.jsonl"
    mpq.save_memory_review_queue(queue, queue_path)
    before = queue_path.read_bytes()

    # Checking an (approved) review queue does not mutate it.
    loaded = mcd.load_proposals_for_check(queue_path)
    assert loaded  # the snapshot is read back as proposals
    mcd.detect_memory_conflicts(
        loaded, mcd.load_existing_memory(_DEMO_EXISTING), now=_NOW)
    assert queue_path.read_bytes() == before


# 16. existing memory file remains byte-identical. ---------------------------

def test_existing_memory_byte_identical_after_check(tmp_path):
    existing_path = tmp_path / "existing.jsonl"
    existing_path.write_bytes(_DEMO_EXISTING.read_bytes())
    before = existing_path.read_bytes()

    proposals = mpq.build_memory_proposals(
        mpq.load_memory_candidates(_DEMO_CANDIDATES))
    mcd.detect_memory_conflicts(
        proposals, mcd.load_existing_memory(existing_path), now=_NOW)
    assert existing_path.read_bytes() == before


# 17. existing v5.0 proposal build is unchanged alongside detection. ---------

def test_v50_proposal_build_unchanged_alongside_detection():
    before = mpq.memory_proposals_to_jsonl(
        mpq.build_memory_proposals(mpq.load_memory_candidates(_DEMO_CANDIDATES)))
    proposals = mpq.build_memory_proposals(
        mpq.load_memory_candidates(_DEMO_CANDIDATES))
    mcd.detect_memory_conflicts(
        proposals, mcd.load_existing_memory(_DEMO_EXISTING), now=_NOW)
    after = mpq.memory_proposals_to_jsonl(
        mpq.build_memory_proposals(mpq.load_memory_candidates(_DEMO_CANDIDATES)))
    assert before == after


# 18. existing v5.1 review queue still works alongside detection. ------------

def test_v51_review_queue_unaffected_by_detection():
    proposals = mpq.build_memory_proposals(
        mpq.load_memory_candidates(_DEMO_CANDIDATES))
    queue = mpq.import_memory_proposals_to_queue(
        [p.to_dict() for p in proposals], [])
    assert all(r.review_status == MemoryReviewStatus.PENDING for r in queue)

    mcd.detect_memory_conflicts(
        proposals, mcd.load_existing_memory(_DEMO_EXISTING), now=_NOW)

    reviewed = mpq.apply_memory_review_to_queue(
        queue, queue[0].proposal_id, MemoryReviewStatus.APPROVED)
    record = {r.proposal_id: r for r in reviewed}[queue[0].proposal_id]
    assert record.review_status == MemoryReviewStatus.APPROVED
    assert record.written is False


# 19. the v3.0 retrieval baseline is unchanged alongside detection. ----------

def test_retrieval_unchanged_alongside_detection(tmp_path, monkeypatch):
    service = _build_service(tmp_path, monkeypatch)
    cases = reh.load_cases(_CASES)

    before = reh.summarize(reh.run_eval(service, cases)).to_dict()
    proposals = mpq.build_memory_proposals(
        mpq.load_memory_candidates(_DEMO_CANDIDATES))
    mcd.detect_memory_conflicts(
        proposals, mcd.load_existing_memory(_DEMO_EXISTING), now=_NOW)
    after = reh.summarize(reh.run_eval(service, cases)).to_dict()
    assert before == after


# README documents the detector purpose and limitations. ---------------------

def test_readme_documents_detector():
    text = _README.read_text(encoding="utf-8")
    assert "v5.2" in text
    assert "does not resolve" in text.lower()


# load_proposals_for_check accepts both a v5.0 export and a v5.1 queue. -------

def test_load_proposals_accepts_export_and_queue(tmp_path):
    proposals = mpq.build_memory_proposals(
        mpq.load_memory_candidates(_DEMO_CANDIDATES))
    export_path = tmp_path / "export.jsonl"
    mpq.write_memory_proposals(proposals, export_path)
    queue = mpq.import_memory_proposals_to_queue(
        [p.to_dict() for p in proposals], [])
    queue_path = tmp_path / "queue.jsonl"
    mpq.save_memory_review_queue(queue, queue_path)

    from_export = mcd.load_proposals_for_check(export_path)
    from_queue = mcd.load_proposals_for_check(queue_path)
    assert {p.proposal_id for p in from_export} \
        == {p.proposal_id for p in from_queue}
    assert all(isinstance(p, MemoryProposalQuality) for p in from_queue)
