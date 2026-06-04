"""Tests for the v4.3 Source Proposal Review Queue.

The review queue lets a human triage the v4.2 maintenance proposals — marking
each approved, rejected, or deferred — *without applying anything*. These tests
pin that contract:

* importing proposals creates ``pending`` review records, and re-importing the
  same proposals is idempotent (no duplicate entries, byte-identical save);
* list output and the CLI are deterministic;
* the documented status transitions hold (pending -> approved/rejected/deferred,
  deferred -> approved/rejected) and disallowed ones fail cleanly with no
  mutation;
* **approved does not mean applied** — ``applied`` stays ``False`` and
  ``applied_at`` stays ``None`` for every record in v4.3;
* reviewing writes *only* the review-queue file: the registry file stays
  byte-identical, knowledge sources are unchanged, and no MemoryLedger entry is
  written;
* the v3.0 retrieval baseline and v2.7 report citation contract are unchanged
  when review runs alongside them;
* the README documents the approved-vs-applied distinction.

Nothing here changes retrieval, ranking, source selection, grounding, composer,
or memory behaviour. A review is a recorded decision, never an applied change.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import source_registry as sr  # noqa: E402
from agent import pack_builder  # noqa: E402
from agent import retrieval_eval_harness as reh  # noqa: E402
from agent.project_packs import PackRegistry  # noqa: E402
from agent.workbench_service import WorkbenchService  # noqa: E402
from retrieval.embedding_backend import OfflineHashingEmbedder  # noqa: E402
from slm.assistant_composer import (  # noqa: E402
    ComposerMode,
    ConsultantReportComposer,
    EvidenceItem,
    GroundingPackage,
    TemplateComposer,
)

_DEMO_REGISTRY = ROOT / "demos" / "source_registry.jsonl"
_MANIFEST = ROOT / "packs" / "m365_coding_assistant" / "pack.yaml"
_CASES = ROOT / "demos" / "retrieval_eval_cases.jsonl"
_README = ROOT / "README.md"

# A fixed "now" so every freshness computation in the suite is deterministic.
_NOW = datetime(2026, 6, 4, tzinfo=timezone.utc)


def _build_service(tmp_path, monkeypatch) -> WorkbenchService:
    """Build the assistant pack into an isolated registry (mirrors other suites)."""
    monkeypatch.chdir(ROOT)
    registry = PackRegistry(tmp_path / "packs")
    plan = pack_builder.PackBuildPlan.from_file(_MANIFEST)
    report = pack_builder.build_pack(plan, registry)
    pack = registry.get_pack(report.pack_id)
    return WorkbenchService.from_pack(
        pack, registry=registry, knowledge_backend="hybrid",
        semantic_embedder=OfflineHashingEmbedder())


def _grounded_package() -> GroundingPackage:
    return GroundingPackage(
        query="how does pydantic validate input",
        mode=ComposerMode.GROUNDED,
        memory_used=False,
        knowledge_used=True,
        model_prior_used=False,
        informational_only=False,
        refused=False,
        route="knowledge",
        evidence=[
            EvidenceItem(citation_id="src:a", kind="knowledge",
                         text="Pydantic validates a typed model on construction.",
                         source_name="doc-a"),
            EvidenceItem(citation_id="src:b", kind="knowledge",
                         text="FastAPI uses Pydantic models to validate bodies.",
                         source_name="doc-b"),
        ],
    )


def _snapshot_tree(root: Path) -> set:
    """Set of file paths under ``root`` (to detect any file written)."""
    return {p for p in root.rglob("*") if p.is_file()}


def _demo_proposal_dicts() -> list[dict]:
    """The v4.2 proposals for the demo registry, as JSONL-style dicts."""
    proposals = sr.propose_source_updates(sr.load_registry(_DEMO_REGISTRY),
                                           now=_NOW)
    return [p.to_dict() for p in proposals]


def _seed_queue(tmp_path) -> tuple[Path, list]:
    """Import the demo proposals into a fresh queue; return (path, queue)."""
    queue_path = tmp_path / "queue.jsonl"
    merged = sr.import_proposals_to_queue(_demo_proposal_dicts(), [])
    sr.save_review_queue(merged, queue_path)
    return queue_path, sr.load_review_queue(queue_path)


# 1. proposals import into pending review records. ---------------------------

def test_proposals_import_into_pending_review_records():
    proposals = _demo_proposal_dicts()
    assert proposals  # the demo registry surfaces maintenance work
    queue = sr.import_proposals_to_queue(proposals, [])

    assert len(queue) == len(proposals)
    imported_ids = {r.proposal_id for r in queue}
    assert imported_ids == {p["proposal_id"] for p in proposals}
    for review in queue:
        assert review.review_status == sr.ReviewStatus.PENDING
        assert review.applied is False
        assert review.applied_at is None
        # Each record preserves a full snapshot of its originating proposal.
        assert review.proposal_snapshot["proposal_id"] == review.proposal_id
        assert review.proposal_snapshot["_record"] == "source_update_proposal"


# 2. duplicate imports are idempotent. ---------------------------------------

def test_duplicate_imports_are_idempotent(tmp_path):
    queue_path, queue = _seed_queue(tmp_path)
    first_bytes = queue_path.read_bytes()

    # Re-import the same proposals on top of the existing queue.
    again = sr.import_proposals_to_queue(_demo_proposal_dicts(), queue)
    assert len(again) == len(queue)
    assert {r.proposal_id for r in again} == {r.proposal_id for r in queue}

    sr.save_review_queue(again, queue_path)
    assert queue_path.read_bytes() == first_bytes


def test_duplicate_import_preserves_existing_review_state():
    proposals = _demo_proposal_dicts()
    queue = sr.import_proposals_to_queue(proposals, [])
    target = queue[0].proposal_id
    reviewed = sr.apply_review_to_queue(queue, target, sr.ReviewStatus.APPROVED)

    # Re-importing must not reset the approved record back to pending.
    merged = sr.import_proposals_to_queue(proposals, reviewed)
    by_id = {r.proposal_id: r for r in merged}
    assert by_id[target].review_status == sr.ReviewStatus.APPROVED


# 3. list output is deterministic. -------------------------------------------

def test_list_output_is_deterministic(tmp_path):
    _, queue = _seed_queue(tmp_path)
    a = sr.render_review_queue_markdown(queue)
    b = sr.render_review_queue_markdown(list(reversed(queue)))
    assert a == b  # sorted by proposal_id regardless of input order
    assert "# Source proposal review queue" in a
    # Queue serialisation is order-independent too.
    assert sr.import_proposals_to_queue(_demo_proposal_dicts(), []) == \
        sr.import_proposals_to_queue(list(reversed(_demo_proposal_dicts())), [])


# 4. pending -> approved. ----------------------------------------------------

def test_pending_to_approved():
    queue = sr.import_proposals_to_queue(_demo_proposal_dicts(), [])
    target = queue[0].proposal_id
    updated = sr.apply_review_to_queue(
        queue, target, sr.ReviewStatus.APPROVED, reviewer="me", note="ok")
    record = {r.proposal_id: r for r in updated}[target]
    assert record.review_status == sr.ReviewStatus.APPROVED
    assert record.reviewer == "me"
    assert record.review_note == "ok"


# 5. pending -> rejected. ----------------------------------------------------

def test_pending_to_rejected():
    queue = sr.import_proposals_to_queue(_demo_proposal_dicts(), [])
    target = queue[0].proposal_id
    updated = sr.apply_review_to_queue(queue, target, sr.ReviewStatus.REJECTED)
    record = {r.proposal_id: r for r in updated}[target]
    assert record.review_status == sr.ReviewStatus.REJECTED


# 6. pending -> deferred. ----------------------------------------------------

def test_pending_to_deferred():
    queue = sr.import_proposals_to_queue(_demo_proposal_dicts(), [])
    target = queue[0].proposal_id
    updated = sr.apply_review_to_queue(queue, target, sr.ReviewStatus.DEFERRED)
    record = {r.proposal_id: r for r in updated}[target]
    assert record.review_status == sr.ReviewStatus.DEFERRED


# 7. deferred -> approved / rejected. ----------------------------------------

def test_deferred_to_approved_and_rejected():
    queue = sr.import_proposals_to_queue(_demo_proposal_dicts(), [])
    target = queue[0].proposal_id

    deferred = sr.apply_review_to_queue(queue, target, sr.ReviewStatus.DEFERRED)
    approved = sr.apply_review_to_queue(deferred, target,
                                        sr.ReviewStatus.APPROVED)
    assert {r.proposal_id: r for r in approved}[target].review_status == \
        sr.ReviewStatus.APPROVED

    rejected = sr.apply_review_to_queue(deferred, target,
                                        sr.ReviewStatus.REJECTED)
    assert {r.proposal_id: r for r in rejected}[target].review_status == \
        sr.ReviewStatus.REJECTED


# 8. invalid status transitions fail cleanly (no mutation). ------------------

def test_invalid_transition_fails_cleanly():
    queue = sr.import_proposals_to_queue(_demo_proposal_dicts(), [])
    target = queue[0].proposal_id

    # rejected is terminal: rejected -> approved is not allowed in v4.3.
    rejected = sr.apply_review_to_queue(queue, target, sr.ReviewStatus.REJECTED)
    rejected_record = {r.proposal_id: r for r in rejected}[target]
    with pytest.raises(ValueError):
        sr.review_proposal(rejected_record, sr.ReviewStatus.APPROVED)

    # approved is terminal too.
    approved = sr.apply_review_to_queue(queue, target, sr.ReviewStatus.APPROVED)
    approved_record = {r.proposal_id: r for r in approved}[target]
    with pytest.raises(ValueError):
        sr.review_proposal(approved_record, sr.ReviewStatus.REJECTED)

    # An unknown status is rejected, and the frozen record is unchanged.
    pending_record = queue[0]
    with pytest.raises(ValueError):
        sr.review_proposal(pending_record, "applied")
    assert pending_record.review_status == sr.ReviewStatus.PENDING

    # A missing proposal_id raises cleanly.
    with pytest.raises(ValueError):
        sr.apply_review_to_queue(queue, "srcprop-does-not-exist",
                                 sr.ReviewStatus.APPROVED)


# 9. approved proposals are not applied. -------------------------------------

def test_approved_proposals_are_not_applied():
    queue = sr.import_proposals_to_queue(_demo_proposal_dicts(), [])
    target = queue[0].proposal_id
    updated = sr.apply_review_to_queue(queue, target, sr.ReviewStatus.APPROVED)
    record = {r.proposal_id: r for r in updated}[target]
    assert record.review_status == sr.ReviewStatus.APPROVED
    assert record.applied is False
    assert record.applied_at is None
    assert record.to_dict()["applied"] is False
    assert record.to_dict()["applied_at"] is None


# 10. applied remains false for every record after any transition. -----------

def test_applied_remains_false_in_v43():
    queue = sr.import_proposals_to_queue(_demo_proposal_dicts(), [])
    # Drive each record through a different decision.
    statuses = [sr.ReviewStatus.APPROVED, sr.ReviewStatus.REJECTED,
                sr.ReviewStatus.DEFERRED]
    for i, review in enumerate(queue):
        queue = sr.apply_review_to_queue(
            queue, review.proposal_id, statuses[i % len(statuses)])
    for review in queue:
        assert review.applied is False
        assert review.applied_at is None


# 11. reviewing writes only the queue file. ----------------------------------

def test_review_updates_only_the_queue_file(tmp_path):
    queue_path, queue = _seed_queue(tmp_path)
    before_tree = _snapshot_tree(tmp_path)
    target = queue[0].proposal_id

    updated = sr.apply_review_to_queue(queue, target, sr.ReviewStatus.APPROVED)
    sr.save_review_queue(updated, queue_path)

    # The only file that exists is still the queue file (no new files written).
    assert _snapshot_tree(tmp_path) == before_tree
    reloaded = {r.proposal_id: r for r in sr.load_review_queue(queue_path)}
    assert reloaded[target].review_status == sr.ReviewStatus.APPROVED


# 12. the registry file is byte-identical after review. ----------------------

def test_registry_file_byte_identical_after_review(tmp_path):
    registry_path = tmp_path / "registry.jsonl"
    registry_path.write_bytes(_DEMO_REGISTRY.read_bytes())
    before = registry_path.read_bytes()

    queue_path = tmp_path / "queue.jsonl"
    proposals = [p.to_dict() for p in sr.propose_source_updates(
        sr.load_registry(registry_path), now=_NOW)]
    queue = sr.import_proposals_to_queue(proposals, [])
    sr.save_review_queue(queue, queue_path)
    queue = sr.apply_review_to_queue(queue, queue[0].proposal_id,
                                     sr.ReviewStatus.APPROVED)
    sr.save_review_queue(queue, queue_path)

    assert registry_path.read_bytes() == before


def test_review_never_calls_save_registry(monkeypatch, tmp_path):
    calls = {"n": 0}
    real_save = sr.save_registry

    def _spy(*args, **kwargs):  # pragma: no cover - should never run
        calls["n"] += 1
        return real_save(*args, **kwargs)

    monkeypatch.setattr(sr, "save_registry", _spy)
    queue_path, queue = _seed_queue(tmp_path)
    updated = sr.apply_review_to_queue(queue, queue[0].proposal_id,
                                       sr.ReviewStatus.APPROVED)
    sr.save_review_queue(updated, queue_path)
    sr.render_review_queue_markdown(updated)
    assert calls["n"] == 0


# 13. source files / knowledge sources are unchanged around review. ----------

def test_knowledge_sources_unchanged_around_review(tmp_path, monkeypatch):
    service = _build_service(tmp_path, monkeypatch)
    before_sources = service.list_knowledge_sources()

    queue_path = tmp_path / "queue.jsonl"
    queue = sr.import_proposals_to_queue(_demo_proposal_dicts(), [])
    sr.save_review_queue(queue, queue_path)
    queue = sr.apply_review_to_queue(queue, queue[0].proposal_id,
                                     sr.ReviewStatus.APPROVED)
    sr.save_review_queue(queue, queue_path)

    assert service.list_knowledge_sources() == before_sources


# 14. no MemoryLedger entry is written by review. ----------------------------

def test_review_writes_no_memory_ledger(tmp_path, monkeypatch):
    service = _build_service(tmp_path, monkeypatch)
    before_count = len(list(service.ledger.entries()))

    queue = sr.import_proposals_to_queue(_demo_proposal_dicts(), [])
    sr.apply_review_to_queue(queue, queue[0].proposal_id,
                             sr.ReviewStatus.APPROVED)

    assert len(list(service.ledger.entries())) == before_count


# 15. the v3.0 retrieval baseline is unchanged alongside review. -------------

def test_retrieval_baseline_unchanged_alongside_review(tmp_path, monkeypatch):
    service = _build_service(tmp_path, monkeypatch)
    queue = sr.import_proposals_to_queue(_demo_proposal_dicts(), [])
    sr.apply_review_to_queue(queue, queue[0].proposal_id,
                             sr.ReviewStatus.APPROVED)

    cases = reh.load_cases(_CASES)
    summary = reh.summarize(reh.run_eval(service, cases))
    assert summary.off_topic_inclusion_rate == 0.0
    assert summary.fail_count == 0


# 16. the report citation contract holds alongside review. -------------------

def test_report_citation_contract_unchanged_alongside_review():
    queue = sr.import_proposals_to_queue(_demo_proposal_dicts(), [])
    sr.apply_review_to_queue(queue, queue[0].proposal_id,
                             sr.ReviewStatus.APPROVED)

    pkg = _grounded_package()
    template = TemplateComposer().compose(pkg)
    report = ConsultantReportComposer().compose(pkg)

    def _citations(answer) -> set:
        import re
        return set(re.findall(r"\[(?:mem|src):[^\]]+\]", answer.text))

    assert _citations(report) == _citations(template)
    assert report.report is not None
    assert template.report is None


# 17. CLI import / list / review is deterministic and queue-only. ------------

def test_cli_import_list_review_deterministic(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "app"))
    import workbench

    proposals_path = tmp_path / "proposals.jsonl"
    sr.write_proposals(
        sr.propose_source_updates(sr.load_registry(_DEMO_REGISTRY), now=_NOW),
        proposals_path)
    queue_path = tmp_path / "queue.jsonl"

    # import-proposals creates the queue file (the only file written).
    before_tree = _snapshot_tree(tmp_path)
    assert workbench.main([
        "source-registry", "proposal-review", "import-proposals",
        str(proposals_path), "--queue", str(queue_path)]) == 0
    after_import = sr.load_review_queue(queue_path)
    assert all(r.review_status == sr.ReviewStatus.PENDING for r in after_import)
    new_files = _snapshot_tree(tmp_path) - before_tree
    assert new_files == {queue_path}
    first_bytes = queue_path.read_bytes()

    # Re-import is idempotent (queue byte-identical).
    assert workbench.main([
        "source-registry", "proposal-review", "import-proposals",
        str(proposals_path), "--queue", str(queue_path)]) == 0
    assert queue_path.read_bytes() == first_bytes

    # list is deterministic and read-only.
    capsys.readouterr()
    assert workbench.main([
        "source-registry", "proposal-review", "list",
        "--queue", str(queue_path)]) == 0
    out_a = capsys.readouterr().out
    assert workbench.main([
        "source-registry", "proposal-review", "list",
        "--queue", str(queue_path)]) == 0
    out_b = capsys.readouterr().out
    assert out_a == out_b
    assert "# Source proposal review queue" in out_a
    assert queue_path.read_bytes() == first_bytes  # list wrote nothing

    # review updates only the queue file.
    target = after_import[0].proposal_id
    tree_before_review = _snapshot_tree(tmp_path)
    assert workbench.main([
        "source-registry", "proposal-review", "review", target,
        "--status", "approved", "--queue", str(queue_path)]) == 0
    assert _snapshot_tree(tmp_path) == tree_before_review  # no new files
    reloaded = {r.proposal_id: r for r in sr.load_review_queue(queue_path)}
    assert reloaded[target].review_status == sr.ReviewStatus.APPROVED
    assert reloaded[target].applied is False


def test_cli_invalid_review_transition_returns_error(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "app"))
    import workbench

    proposals_path = tmp_path / "proposals.jsonl"
    sr.write_proposals(
        sr.propose_source_updates(sr.load_registry(_DEMO_REGISTRY), now=_NOW),
        proposals_path)
    queue_path = tmp_path / "queue.jsonl"
    assert workbench.main([
        "source-registry", "proposal-review", "import-proposals",
        str(proposals_path), "--queue", str(queue_path)]) == 0
    target = sr.load_review_queue(queue_path)[0].proposal_id

    assert workbench.main([
        "source-registry", "proposal-review", "review", target,
        "--status", "rejected", "--queue", str(queue_path)]) == 0
    before = queue_path.read_bytes()
    # rejected -> approved is invalid; the CLI fails cleanly and writes nothing.
    assert workbench.main([
        "source-registry", "proposal-review", "review", target,
        "--status", "approved", "--queue", str(queue_path)]) == 1
    assert queue_path.read_bytes() == before


# 18. the README documents the approved-vs-applied distinction. --------------

def test_readme_documents_approved_vs_applied():
    text = _README.read_text(encoding="utf-8")
    assert "v4.3" in text
    assert "Approved does not mean applied" in text
    assert "proposal-review" in text
