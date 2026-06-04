"""Tests for the v5.1 Memory Proposal Review Queue.

The review queue lets a human triage the v5.0 memory proposals — marking each
approved, rejected, or deferred — *without writing any memory*. These tests pin
that contract:

* importing proposals creates ``pending`` review records, and re-importing the
  same proposals is idempotent (no duplicate entries, byte-identical save) while
  preserving any existing approved/rejected/deferred state;
* list output and the CLI are deterministic;
* the documented status transitions hold (pending -> approved/rejected/deferred,
  deferred -> approved/rejected) and disallowed ones fail cleanly with no
  mutation;
* **approved does not mean written** — ``written`` stays ``False`` and
  ``written_at`` stays ``None`` for every record in v5.1;
* reviewing writes *only* the review-queue file: the proposals file stays
  byte-identical and no ``MemoryLedger`` entry is written;
* the full proposal snapshot is preserved on every queue entry;
* the v5.0 memory-proposal quality build, the v4.3 source review queue, and the
  v3.0 retrieval baseline are unchanged when memory review runs alongside them;
* the README documents the approved-vs-written distinction.

Nothing here changes retrieval, ranking, source selection, grounding, composer,
or memory behaviour. A review is a recorded decision, never a written memory.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import memory_proposal_quality as mpq  # noqa: E402
from agent.memory_proposal_quality import MemoryReviewStatus  # noqa: E402
from agent import source_registry as sr  # noqa: E402
from agent import pack_builder  # noqa: E402
from agent import retrieval_eval_harness as reh  # noqa: E402
from agent.project_packs import PackRegistry  # noqa: E402
from agent.workbench_service import WorkbenchService  # noqa: E402
from agent.memory_ledger import MemoryLedger  # noqa: E402
from retrieval.embedding_backend import OfflineHashingEmbedder  # noqa: E402

_DEMO = ROOT / "demos" / "memory_proposal_candidates.jsonl"
_DEMO_REGISTRY = ROOT / "demos" / "source_registry.jsonl"
_MANIFEST = ROOT / "packs" / "m365_coding_assistant" / "pack.yaml"
_CASES = ROOT / "demos" / "retrieval_eval_cases.jsonl"
_README = ROOT / "README.md"


def _snapshot_tree(root: Path) -> set:
    """Set of file paths under ``root`` (to detect any file written)."""
    return {p for p in root.rglob("*") if p.is_file()}


def _demo_proposal_dicts() -> list[dict]:
    """The v5.0 memory proposals for the demo candidates, as JSONL-style dicts."""
    proposals = mpq.build_memory_proposals(mpq.load_memory_candidates(_DEMO))
    return [p.to_dict() for p in proposals]


def _seed_queue(tmp_path) -> tuple[Path, list]:
    """Import the demo proposals into a fresh queue; return (path, queue)."""
    queue_path = tmp_path / "queue.jsonl"
    merged = mpq.import_memory_proposals_to_queue(_demo_proposal_dicts(), [])
    mpq.save_memory_review_queue(merged, queue_path)
    return queue_path, mpq.load_memory_review_queue(queue_path)


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


# 1. proposals import into pending review records. ---------------------------

def test_proposals_import_into_pending_review_records():
    proposals = _demo_proposal_dicts()
    assert proposals  # the demo candidates produce proposals
    queue = mpq.import_memory_proposals_to_queue(proposals, [])

    assert len(queue) == len(proposals)
    assert {r.proposal_id for r in queue} == {p["proposal_id"] for p in proposals}
    assert all(r.review_status == MemoryReviewStatus.PENDING for r in queue)
    assert all(r.written is False for r in queue)
    assert all(r.written_at is None for r in queue)


# 2. duplicate import is idempotent. -----------------------------------------

def test_duplicate_import_is_idempotent(tmp_path):
    proposals = _demo_proposal_dicts()
    queue = mpq.import_memory_proposals_to_queue(proposals, [])
    path = tmp_path / "queue.jsonl"
    mpq.save_memory_review_queue(queue, path)
    first_bytes = path.read_bytes()

    # Re-importing the same proposals adds nothing and saves byte-identically.
    again = mpq.import_memory_proposals_to_queue(proposals, queue)
    assert len(again) == len(queue)
    mpq.save_memory_review_queue(again, path)
    assert path.read_bytes() == first_bytes


# 3. re-import preserves existing approved/rejected/deferred state. ----------

def test_reimport_preserves_existing_review_state():
    proposals = _demo_proposal_dicts()
    queue = mpq.import_memory_proposals_to_queue(proposals, [])
    ids = [r.proposal_id for r in queue]

    queue = mpq.apply_memory_review_to_queue(
        queue, ids[0], MemoryReviewStatus.APPROVED)
    queue = mpq.apply_memory_review_to_queue(
        queue, ids[1], MemoryReviewStatus.REJECTED)
    queue = mpq.apply_memory_review_to_queue(
        queue, ids[2], MemoryReviewStatus.DEFERRED)

    reimported = mpq.import_memory_proposals_to_queue(proposals, queue)
    by_id = {r.proposal_id: r for r in reimported}
    assert by_id[ids[0]].review_status == MemoryReviewStatus.APPROVED
    assert by_id[ids[1]].review_status == MemoryReviewStatus.REJECTED
    assert by_id[ids[2]].review_status == MemoryReviewStatus.DEFERRED
    # Every other record stays pending; nothing duplicated.
    assert len(reimported) == len(proposals)


# 4. list output is deterministic. -------------------------------------------

def test_list_output_is_deterministic():
    queue = mpq.import_memory_proposals_to_queue(_demo_proposal_dicts(), [])
    out_a = mpq.render_memory_review_queue_markdown(queue)
    out_b = mpq.render_memory_review_queue_markdown(queue)
    assert out_a == out_b
    assert "# Memory proposal review queue" in out_a
    assert "approved does not mean written" in out_a
    assert "written: 0 (always 0 in v5.1)" in out_a


# 5. pending -> approved. ----------------------------------------------------

def test_pending_can_transition_to_approved():
    queue = mpq.import_memory_proposals_to_queue(_demo_proposal_dicts(), [])
    target = queue[0].proposal_id
    updated = mpq.apply_memory_review_to_queue(
        queue, target, MemoryReviewStatus.APPROVED, reviewer="alex",
        note="useful")
    record = {r.proposal_id: r for r in updated}[target]
    assert record.review_status == MemoryReviewStatus.APPROVED
    assert record.reviewer == "alex"
    assert record.review_note == "useful"
    assert record.written is False
    assert record.written_at is None


# 6. pending -> rejected. ----------------------------------------------------

def test_pending_can_transition_to_rejected():
    queue = mpq.import_memory_proposals_to_queue(_demo_proposal_dicts(), [])
    target = queue[0].proposal_id
    updated = mpq.apply_memory_review_to_queue(
        queue, target, MemoryReviewStatus.REJECTED)
    record = {r.proposal_id: r for r in updated}[target]
    assert record.review_status == MemoryReviewStatus.REJECTED
    assert record.written is False


# 7. pending -> deferred. ----------------------------------------------------

def test_pending_can_transition_to_deferred():
    queue = mpq.import_memory_proposals_to_queue(_demo_proposal_dicts(), [])
    target = queue[0].proposal_id
    updated = mpq.apply_memory_review_to_queue(
        queue, target, MemoryReviewStatus.DEFERRED)
    record = {r.proposal_id: r for r in updated}[target]
    assert record.review_status == MemoryReviewStatus.DEFERRED
    assert record.written is False


# 8. deferred -> approved / rejected. ----------------------------------------

def test_deferred_can_transition_to_approved_or_rejected():
    queue = mpq.import_memory_proposals_to_queue(_demo_proposal_dicts(), [])
    a_id, b_id = queue[0].proposal_id, queue[1].proposal_id

    queue = mpq.apply_memory_review_to_queue(
        queue, a_id, MemoryReviewStatus.DEFERRED)
    queue = mpq.apply_memory_review_to_queue(
        queue, b_id, MemoryReviewStatus.DEFERRED)

    queue = mpq.apply_memory_review_to_queue(
        queue, a_id, MemoryReviewStatus.APPROVED)
    queue = mpq.apply_memory_review_to_queue(
        queue, b_id, MemoryReviewStatus.REJECTED)

    by_id = {r.proposal_id: r for r in queue}
    assert by_id[a_id].review_status == MemoryReviewStatus.APPROVED
    assert by_id[b_id].review_status == MemoryReviewStatus.REJECTED
    assert by_id[a_id].written is False
    assert by_id[b_id].written is False


# 9. invalid transition fails cleanly (no mutation). -------------------------

def test_invalid_transition_fails_cleanly():
    queue = mpq.import_memory_proposals_to_queue(_demo_proposal_dicts(), [])
    target = queue[0].proposal_id
    queue = mpq.apply_memory_review_to_queue(
        queue, target, MemoryReviewStatus.APPROVED)
    settled = {r.proposal_id: r for r in queue}[target]

    # approved is terminal: approved -> rejected is not allowed.
    with pytest.raises(ValueError):
        mpq.apply_memory_review_to_queue(
            queue, target, MemoryReviewStatus.REJECTED)
    # An unknown status is rejected too.
    with pytest.raises(ValueError):
        mpq.apply_memory_review_to_queue(queue, target, "archived")
    # A missing proposal_id is rejected.
    with pytest.raises(ValueError):
        mpq.apply_memory_review_to_queue(
            queue, "memprop-does-not-exist", MemoryReviewStatus.APPROVED)

    # The settled record is untouched by the failed transitions.
    still = {r.proposal_id: r for r in queue}[target]
    assert still == settled
    assert still.review_status == MemoryReviewStatus.APPROVED


# 10 & 11. approved proposals are not written; written stays false. ----------

def test_approved_proposals_are_not_written():
    queue = mpq.import_memory_proposals_to_queue(_demo_proposal_dicts(), [])
    for target in [r.proposal_id for r in queue]:
        if mpq.memory_review_status_transition_allowed(
                MemoryReviewStatus.PENDING, MemoryReviewStatus.APPROVED):
            queue = mpq.apply_memory_review_to_queue(
                queue, target, MemoryReviewStatus.APPROVED)

    # Every record is approved, yet none is written in v5.1.
    assert all(r.review_status == MemoryReviewStatus.APPROVED for r in queue)
    assert all(r.written is False for r in queue)
    assert all(r.written_at is None for r in queue)


def test_written_remains_false_through_full_lifecycle():
    queue = mpq.import_memory_proposals_to_queue(_demo_proposal_dicts(), [])
    a_id, b_id, c_id = (queue[0].proposal_id, queue[1].proposal_id,
                        queue[2].proposal_id)
    queue = mpq.apply_memory_review_to_queue(
        queue, a_id, MemoryReviewStatus.APPROVED)
    queue = mpq.apply_memory_review_to_queue(
        queue, b_id, MemoryReviewStatus.DEFERRED)
    queue = mpq.apply_memory_review_to_queue(
        queue, b_id, MemoryReviewStatus.APPROVED)
    queue = mpq.apply_memory_review_to_queue(
        queue, c_id, MemoryReviewStatus.REJECTED)
    assert all(r.written is False and r.written_at is None for r in queue)


# 12. MemoryLedger.add is never called by the review flow. -------------------

def test_review_flow_never_calls_memory_ledger_add(tmp_path, monkeypatch):
    calls = []
    original_add = MemoryLedger.add

    def _spy(self, *args, **kwargs):
        calls.append(args)
        return original_add(self, *args, **kwargs)

    monkeypatch.setattr(MemoryLedger, "add", _spy)

    queue = mpq.import_memory_proposals_to_queue(_demo_proposal_dicts(), [])
    queue = mpq.apply_memory_review_to_queue(
        queue, queue[0].proposal_id, MemoryReviewStatus.APPROVED)
    mpq.save_memory_review_queue(queue, tmp_path / "queue.jsonl")
    mpq.render_memory_review_queue_markdown(queue)

    assert calls == []


# 13. review updates only the queue file. ------------------------------------

def test_review_updates_only_the_queue_file(tmp_path):
    proposals_path = tmp_path / "proposals.jsonl"
    mpq.write_memory_proposals(
        mpq.build_memory_proposals(mpq.load_memory_candidates(_DEMO)),
        proposals_path)
    proposals_before = proposals_path.read_bytes()

    queue_path = tmp_path / "queue.jsonl"
    merged = mpq.import_memory_proposals_to_queue(
        mpq.load_memory_proposal_dicts(proposals_path), [])
    mpq.save_memory_review_queue(merged, queue_path)

    tree_before = _snapshot_tree(tmp_path)
    target = mpq.load_memory_review_queue(queue_path)[0].proposal_id
    updated = mpq.apply_memory_review_to_queue(
        mpq.load_memory_review_queue(queue_path), target,
        MemoryReviewStatus.APPROVED)
    mpq.save_memory_review_queue(updated, queue_path)

    # No new files were created, and the proposals file is byte-identical.
    assert _snapshot_tree(tmp_path) == tree_before
    assert proposals_path.read_bytes() == proposals_before


# 14. the full proposal snapshot is preserved on every queue entry. ----------

def test_proposal_snapshot_is_preserved():
    proposals = _demo_proposal_dicts()
    queue = mpq.import_memory_proposals_to_queue(proposals, [])
    by_id = {r.proposal_id: r for r in queue}
    for proposal in proposals:
        snap = by_id[proposal["proposal_id"]].proposal_snapshot
        assert snap == proposal

    # The snapshot survives a save -> load round-trip and a review transition.
    target = queue[0].proposal_id
    expected = by_id[target].proposal_snapshot
    reviewed = mpq.apply_memory_review_to_queue(
        queue, target, MemoryReviewStatus.APPROVED)
    assert {r.proposal_id: r for r in reviewed}[target].proposal_snapshot \
        == expected


# 15. the v5.0 memory-proposal quality build is unchanged alongside review. --

def test_v50_proposal_build_unchanged_alongside_review():
    before = mpq.memory_proposals_to_jsonl(
        mpq.build_memory_proposals(mpq.load_memory_candidates(_DEMO)))

    queue = mpq.import_memory_proposals_to_queue(_demo_proposal_dicts(), [])
    mpq.apply_memory_review_to_queue(
        queue, queue[0].proposal_id, MemoryReviewStatus.APPROVED)

    after = mpq.memory_proposals_to_jsonl(
        mpq.build_memory_proposals(mpq.load_memory_candidates(_DEMO)))
    assert before == after


# 16. the v4.3 source proposal review queue is unaffected. -------------------

def test_source_proposal_review_queue_still_works():
    from datetime import datetime, timezone
    now = datetime(2026, 6, 4, tzinfo=timezone.utc)
    src_proposals = [
        p.to_dict() for p in sr.propose_source_updates(
            sr.load_registry(_DEMO_REGISTRY), now=now)]
    queue = sr.import_proposals_to_queue(src_proposals, [])
    assert all(r.review_status == sr.ReviewStatus.PENDING for r in queue)
    reviewed = sr.apply_review_to_queue(
        queue, queue[0].proposal_id, sr.ReviewStatus.APPROVED)
    assert {r.proposal_id: r for r in reviewed}[
        queue[0].proposal_id].applied is False


# 17. the v3.0 retrieval baseline is unchanged alongside memory review. ------

def test_retrieval_unchanged_alongside_memory_review(tmp_path, monkeypatch):
    service = _build_service(tmp_path, monkeypatch)
    cases = reh.load_cases(_CASES)

    before = reh.summarize(reh.run_eval(service, cases)).to_dict()
    queue = mpq.import_memory_proposals_to_queue(_demo_proposal_dicts(), [])
    mpq.apply_memory_review_to_queue(
        queue, queue[0].proposal_id, MemoryReviewStatus.APPROVED)
    after = reh.summarize(reh.run_eval(service, cases)).to_dict()

    assert before == after


# 18. README documents the approved-vs-written distinction. ------------------

def test_readme_documents_approved_vs_written():
    text = _README.read_text(encoding="utf-8")
    assert "v5.1" in text
    assert "approved does not mean written" in text.lower()


# CLI: import / list / review is deterministic and queue-only. ---------------

def test_cli_memory_review_import_list_review(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "app"))
    import workbench

    proposals_path = tmp_path / "proposals.jsonl"
    mpq.write_memory_proposals(
        mpq.build_memory_proposals(mpq.load_memory_candidates(_DEMO)),
        proposals_path)
    queue_path = tmp_path / "queue.jsonl"

    # review-import creates the queue file (the only file written).
    before_tree = _snapshot_tree(tmp_path)
    assert workbench.main([
        "memory-proposals", "review-import", str(proposals_path),
        "--queue", str(queue_path)]) == 0
    after_import = mpq.load_memory_review_queue(queue_path)
    assert after_import
    assert all(r.review_status == MemoryReviewStatus.PENDING
               for r in after_import)
    assert _snapshot_tree(tmp_path) - before_tree == {queue_path}
    first_bytes = queue_path.read_bytes()

    # Re-import is idempotent (queue byte-identical).
    assert workbench.main([
        "memory-proposals", "review-import", str(proposals_path),
        "--queue", str(queue_path)]) == 0
    assert queue_path.read_bytes() == first_bytes

    # review-list is deterministic and read-only.
    capsys.readouterr()
    assert workbench.main([
        "memory-proposals", "review-list", "--queue", str(queue_path)]) == 0
    out_a = capsys.readouterr().out
    assert workbench.main([
        "memory-proposals", "review-list", "--queue", str(queue_path)]) == 0
    out_b = capsys.readouterr().out
    assert out_a == out_b
    assert "# Memory proposal review queue" in out_a
    assert queue_path.read_bytes() == first_bytes  # list wrote nothing

    # review updates only the queue file.
    target = after_import[0].proposal_id
    tree_before_review = _snapshot_tree(tmp_path)
    assert workbench.main([
        "memory-proposals", "review", target, "--status", "approved",
        "--queue", str(queue_path)]) == 0
    assert _snapshot_tree(tmp_path) == tree_before_review  # no new files
    reloaded = {r.proposal_id: r for r in mpq.load_memory_review_queue(queue_path)}
    assert reloaded[target].review_status == MemoryReviewStatus.APPROVED
    assert reloaded[target].written is False


def test_cli_memory_review_invalid_transition_returns_error(tmp_path,
                                                            monkeypatch):
    monkeypatch.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "app"))
    import workbench

    proposals_path = tmp_path / "proposals.jsonl"
    mpq.write_memory_proposals(
        mpq.build_memory_proposals(mpq.load_memory_candidates(_DEMO)),
        proposals_path)
    queue_path = tmp_path / "queue.jsonl"
    assert workbench.main([
        "memory-proposals", "review-import", str(proposals_path),
        "--queue", str(queue_path)]) == 0
    target = mpq.load_memory_review_queue(queue_path)[0].proposal_id

    # Settle it as rejected (terminal).
    assert workbench.main([
        "memory-proposals", "review", target, "--status", "rejected",
        "--queue", str(queue_path)]) == 0
    before = queue_path.read_bytes()

    # rejected -> approved is invalid: exit 1, queue unwritten.
    assert workbench.main([
        "memory-proposals", "review", target, "--status", "approved",
        "--queue", str(queue_path)]) == 1
    assert queue_path.read_bytes() == before
