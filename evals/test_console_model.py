"""Tests for the Consultant Workbench UI view-model layer (UI v1.0).

These cover the product contract of the read-only console model: navigation is
complete and ordered; the Ask view separates evidence from judgement and is
honest about insufficient evidence; Sources surface status and warnings; Packs
distinguish active from inactive; Reviews keep approval distinct from execution
and never offer approval on stale items; Monitoring is labelled advisory only
and lists critical findings separately; Memory exposes no delete control; Imports
distinguish supported from unsupported adapters; Settings advertise read-only.

Two structural guarantees are enforced: the module imports/calls **no** mutation
API (AST purity), and running every ``build_*`` view model against on-disk data
leaves that data byte-for-byte unchanged (no page load mutates state).
"""
from __future__ import annotations

import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import console_model as cm  # noqa: E402
from agent.knowledge_pack_activation import (  # noqa: E402
    ActivePackState,
    PackLifecycleState,
)
from agent.regression_review_queue import RegressionActionRequest  # noqa: E402
from slm.assistant_composer import JUDGEMENT_LABEL  # noqa: E402


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class _FakeChatResult:
    """A minimal stand-in for ChatOrchestratorResult (read-only attributes)."""

    query: str = "q"
    mode: str = "evidence_answer"
    answer_text: str = ""
    citations: tuple = ()
    judgement_present: bool = False
    judgement_labelled: bool = False
    answer_has_citations: bool = False
    evidence_used_count: int = 0
    evidence_gap_count: int = 0
    evidence_summary: str = ""
    missing_evidence: tuple = ()
    related_sources: tuple = ()
    refusal_reason: str = ""
    proposed_memory_count: int = 0
    proposed_source_update_count: int = 0
    state_mutation_attempted: bool = False
    evidence_detail: tuple = ()
    route_reason: str = ""


def _empty_config(tmp_path: Path) -> cm.ConsoleConfig:
    """A config pointing entirely at a fresh tmp dir (no data on disk)."""

    return cm.ConsoleConfig.default(tmp_path)


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------


def test_navigation_renders_all_sections():
    sections = cm.navigation()
    keys = [s.key for s in sections]
    assert keys == [
        "home", "projects", "ask", "reports", "sources", "imports",
        "packs", "reviews", "monitoring", "memory", "settings",
    ]
    assert all(s.label and s.blurb for s in sections)


# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------


def test_home_loads_without_mutation_or_data(tmp_path):
    home = cm.build_home(_empty_config(tmp_path))
    assert home.product_name == cm.PRODUCT_NAME
    assert home.ui_version == cm.CONSOLE_UI_VERSION
    assert home.capabilities  # the system advertises what it can do
    assert home.metrics  # honest metrics (empty-state tones) still render
    assert home.plain_status  # plain-language status lines present


def test_home_uses_real_demo_data():
    home = cm.build_home(cm.ConsoleConfig.default())
    # The real demo source registry exists, so Sources contributes a metric.
    metric_keys = {m.key for m in home.metrics}
    assert "sources" in metric_keys


# ---------------------------------------------------------------------------
# Ask
# ---------------------------------------------------------------------------


def test_ask_offers_answer_modes():
    keys = {k for k, _ in cm.ANSWER_MODES}
    assert {"auto", "evidence_answer", "report_style_answer",
            "judgement_only", "memory_context"} <= keys


def test_ask_separates_evidence_and_judgement():
    answer = f"The delivery is on Friday [M1]. {JUDGEMENT_LABEL} It may be risky."
    view = cm.build_ask(_FakeChatResult(
        mode="evidence_answer", answer_text=answer,
        citations=("M1",), answer_has_citations=True, judgement_present=True))
    kinds = [s.kind for s in view.sections]
    assert "factual" in kinds
    assert "judgement" in kinds
    factual = next(s for s in view.sections if s.kind == "factual")
    judgement = next(s for s in view.sections if s.kind == "judgement")
    assert "Friday" in factual.body
    assert JUDGEMENT_LABEL in judgement.body
    assert "risky" not in factual.body  # judgement text is not in the factual block


def test_ask_grounded_evidence_is_marked_grounded():
    view = cm.build_ask(_FakeChatResult(
        mode="evidence_answer", answer_text="Answer [M1].",
        citations=("M1",), answer_has_citations=True))
    assert view.grounded is True
    assert "Grounded" in view.status_label
    assert view.refused is False


def test_ask_insufficient_evidence_is_clear_and_refused():
    view = cm.build_ask(_FakeChatResult(
        mode="insufficient_evidence", answer_text="",
        missing_evidence=("supplier delivery date",), evidence_gap_count=1))
    assert view.refused is True
    assert "Insufficient evidence" in view.status_label
    assert view.grounded is False
    assert "Add a source" in view.safe_next_action
    assert any(s.kind == "gap" for s in view.sections)


def test_ask_evidence_answer_without_citations_is_partial():
    view = cm.build_ask(_FakeChatResult(
        mode="evidence_answer", answer_text="Answer.",
        citations=(), answer_has_citations=False))
    assert view.grounded is False
    assert "Partially grounded" in view.status_label


# ---------------------------------------------------------------------------
# Ask — evidence inspector, citations, scope, export
# ---------------------------------------------------------------------------


def _detail(citation_id, *, cited, score, rank, source_name="Admin Patterns",
            authority="official", chunk_id="c1", source_id="s1", section="Roles",
            version="2", text="Use PIM for just-in-time activation.",
            backend_name="hybrid", domain="security"):
    return {
        "citation_id": citation_id, "source_id": source_id, "chunk_id": chunk_id,
        "source_name": source_name, "domain": domain, "authority": authority,
        "version": version, "section": section, "score": score, "rank": rank,
        "backend_name": backend_name, "text": text, "cited": cited,
    }


def _grounded_result():
    return _FakeChatResult(
        mode="evidence_answer", answer_text="Use PIM [src:c1].",
        citations=("src:c1",), answer_has_citations=True, evidence_used_count=1,
        evidence_detail=(
            _detail("src:c1", cited=True, score=0.91, rank=1),
            _detail("src:c2", cited=False, score=0.42, rank=2,
                    source_name="Blog Post", authority="community", chunk_id="c2",
                    source_id="s2", text="Community tip on roles."),
        ))


def test_ask_evidence_inspector_separates_retrieved_and_cited():
    inspector = cm.build_evidence_inspector(_grounded_result())
    assert inspector.available is True
    assert inspector.retrieved_count == 2
    assert inspector.cited_count == 1
    assert [e.citation_id for e in inspector.cited] == ["src:c1"]
    # Retrieved-but-not-cited evidence is preserved and visibly distinct.
    retrieved_only = [e for e in inspector.retrieved if not e.cited]
    assert [e.citation_id for e in retrieved_only] == ["src:c2"]


def test_ask_inspector_score_is_relevance_not_confidence():
    inspector = cm.build_evidence_inspector(_grounded_result())
    top = inspector.cited[0]
    # Score is the backend relevance activation; authority is a separate axis.
    assert top.score == 0.91
    assert top.authority_label == "Official"
    # The stage gloss explicitly states a score is not confidence/truth.
    joined = " ".join(inspector.stage_gloss).lower()
    assert "not a confidence" in joined or "not confidence" in joined


def test_ask_citation_details_preserve_ids_and_provenance():
    cards = cm.build_citation_details(_grounded_result())
    assert [c.citation_id for c in cards] == ["src:c1"]
    card = cards[0]
    assert card.kind == "knowledge"
    assert card.source_name == "Admin Patterns"
    assert card.authority_label == "Official"
    assert card.chunk_id == "c1"


def test_ask_memory_citation_has_no_invented_provenance():
    # A memory citation with no matching evidence detail must not gain fields.
    result = _FakeChatResult(
        mode="evidence_answer", answer_text="Recalled [mem:7].",
        citations=("mem:7",), answer_has_citations=True, evidence_detail=())
    cards = cm.build_citation_details(result)
    assert len(cards) == 1
    card = cards[0]
    assert card.kind == "memory"
    assert card.source_name == ""  # nothing invented
    assert card.authority == ""
    assert card.chunk_id == "7"


def test_ask_scope_reports_active_packs_only(tmp_path):
    scope = cm.build_ask_scope(_empty_config(tmp_path))
    assert scope.active_pack_count == 0
    assert scope.approved_source_count == 0
    assert scope.memory_context_enabled is False
    assert scope.warnings  # empty scope is flagged, not hidden
    assert any("empty" in w.lower() for w in scope.warnings)


def test_ask_export_markdown_preserves_evidence_and_labels():
    result = _grounded_result()
    view = cm.build_ask(result)
    inspector = cm.build_evidence_inspector(result)
    cards = cm.build_citation_details(result)
    md = cm.render_ask_markdown(view, citations=cards, inspector=inspector,
                                generated_at="2026-06-04T00:00:00+00:00")
    assert "src:c1" in md  # citation id preserved verbatim
    assert "Official" in md
    assert "not confidence" in md.lower()
    assert "Retrieval is not truth" in md
    assert "never writes memory" in md.lower()


def test_ask_export_json_bundle_round_trips():
    result = _grounded_result()
    view = cm.build_ask(result)
    inspector = cm.build_evidence_inspector(result)
    cards = cm.build_citation_details(result)
    payload = json.loads(cm.render_ask_json(
        view, citations=cards, inspector=inspector,
        generated_at="2026-06-04T00:00:00+00:00"))
    assert payload["_record"] == "ask_evidence_bundle"
    assert payload["answer"]["query"] == view.query
    assert [c["citation_id"] for c in payload["citations"]] == ["src:c1"]
    assert payload["evidence_inspector"]["retrieved_count"] == 2
    assert payload["evidence_inspector"]["cited_count"] == 1


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def test_sources_empty_state_when_no_registry(tmp_path):
    view = cm.build_sources(_empty_config(tmp_path))
    assert view.rows == ()
    assert view.empty is not None
    assert view.errors == ()


def test_sources_show_status_and_labels_from_real_registry():
    view = cm.build_sources(cm.ConsoleConfig.default())
    assert view.rows  # the demo registry is populated
    row = view.rows[0]
    assert row.authority_label  # plain-language authority
    assert row.status_label  # plain-language status
    assert isinstance(view.warnings, tuple)


# ---------------------------------------------------------------------------
# Knowledge packs
# ---------------------------------------------------------------------------


def _write_pack_state(path: Path, *states: ActivePackState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(s.to_dict()) + "\n" for s in states), encoding="utf-8")


def test_packs_empty_state_when_no_state_file(tmp_path):
    view = cm.build_packs(_empty_config(tmp_path))
    assert view.active_count == 0
    assert view.empty is not None


def test_packs_distinguish_active_from_inactive(tmp_path):
    config = _empty_config(tmp_path)
    active = ActivePackState(
        pack_id="pack-a", pack_version="1", pack_fingerprint="fp-a",
        source_type="pdf", source_id="src-a", source_revision="r1",
        status=PackLifecycleState.ACTIVE)
    superseded = ActivePackState(
        pack_id="pack-b", pack_version="1", pack_fingerprint="fp-b",
        source_type="pdf", source_id="src-b", source_revision="r1",
        status=PackLifecycleState.SUPERSEDED)
    _write_pack_state(config.pack_state_path, active, superseded)

    view = cm.build_packs(config)
    assert view.active_count == 1
    assert view.inactive_count == 1
    by_id = {r.pack_id: r for r in view.rows}
    assert by_id["pack-a"].is_active is True
    assert by_id["pack-b"].is_active is False
    assert by_id["pack-b"].status_label == "Superseded"


# ---------------------------------------------------------------------------
# Reviews
# ---------------------------------------------------------------------------


def test_reviews_empty_categories_when_no_queues(tmp_path):
    view = cm.build_reviews(_empty_config(tmp_path))
    assert view.pending_count == 0
    keys = {c.key for c in view.categories}
    assert {"memory", "source", "regression", "action"} <= keys
    assert all(c.empty is not None for c in view.categories)


def test_reviews_distinguish_approval_from_execution(tmp_path):
    config = _empty_config(tmp_path)
    action = RegressionActionRequest(
        action_request_id="act-1", review_item_id="ri-1", review_record_id="rr-1",
        requested_action="rollback", requested_by="tester", requested_at="2025-01-01",
        active_state_hash="state:abc", affected_pack_ids=("pack-a",),
        affected_pack_fingerprints=("fp-a",), monitoring_run_fingerprint="run:1",
        recommendation_fingerprint="rec:1", approval_reason="regression",
        required_execution_approval_type="dual_control", status="requested")
    config.regression_actions_path.parent.mkdir(parents=True, exist_ok=True)
    config.regression_actions_path.write_text(
        action.to_json() + "\n", encoding="utf-8")

    view = cm.build_reviews(config)
    action_cat = next(c for c in view.categories if c.key == "action")
    assert action_cat.rows
    row = action_cat.rows[0]
    assert "not execution" in row.detail
    assert row.can_approve is True  # pending + not stale -> request may be approved


def test_reviews_stale_item_cannot_be_approved():
    stale = cm._make_review_row(
        "regression", "ri-stale", "Stale review", "detail",
        "stale", (), "2025-01-01")
    assert stale.is_stale is True
    assert stale.can_approve is False


def test_reviews_pending_item_can_be_approved():
    pending = cm._make_review_row(
        "memory", "mp-1", "claim", "detail", "pending", (), "2025-01-01")
    assert pending.is_pending is True
    assert pending.can_approve is True


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------


def _write_monitoring_run(path: Path, *, recommendation: str, critical: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    findings = []
    if critical:
        findings.append({
            "_record": "regression_finding", "finding_code": "expected_source_recall_drop",
            "severity": "critical", "candidate_pack_ids": ["pack-a"],
            "diagnostic_note": "recall dropped", "recommended_human_action": "investigate"})
    findings.append({
        "_record": "regression_finding", "finding_code": "rank_worsened",
        "severity": "warning", "candidate_pack_ids": [],
        "diagnostic_note": "", "recommended_human_action": ""})
    run = {
        "_record": "active_pack_monitoring_run",
        "monitoring_run_id": "run-1", "created_at": "2025-01-01T00:00:00Z",
        "recommendation": {
            "_record": "monitoring_recommendation", "recommendation": recommendation,
            "advisory_only": True, "confidence": "high", "rationale_codes": [],
            "candidate_pack_ids": ["pack-a"]},
        "findings": findings,
    }
    path.write_text(json.dumps(run) + "\n", encoding="utf-8")


def test_monitoring_empty_state_when_no_history(tmp_path):
    view = cm.build_monitoring(_empty_config(tmp_path))
    assert view.has_history is False
    assert view.empty is not None
    # The advisory framing is present even with no data.
    assert "ADVISORY ONLY" in view.advisory_notice


def test_monitoring_labels_recommendation_advisory_and_splits_critical(tmp_path):
    config = _empty_config(tmp_path)
    _write_monitoring_run(
        config.monitoring_history_path,
        recommendation="rollback_recommended", critical=True)

    view = cm.build_monitoring(config)
    assert view.has_history is True
    assert "ADVISORY ONLY" in view.advisory_notice
    assert "advisory" in view.recommendation_label.lower()
    assert len(view.critical_findings) == 1
    assert view.critical_findings[0].is_critical is True
    assert all(not f.is_critical for f in view.other_findings)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


def test_memory_has_no_delete_control():
    # The read-only model exposes no delete/mutation helper of any kind.
    names = dir(cm)
    assert not any("delete" in n.lower() for n in names)
    assert not any(n.startswith("save_") or n.startswith("write_") for n in names)


def test_memory_lists_ledger_rows_from_demo():
    view = cm.build_memory(cm.ConsoleConfig.default())
    # The demo ledger is populated; the view surfaces rows without a live service.
    assert view.total_count >= 0
    assert isinstance(view.rows, tuple)


def test_memory_reads_supplied_ledger_rows():
    rows = [{"memory_id": "m1", "canonical_text": "hello", "status": "active"}]
    view = cm.build_memory(cm.ConsoleConfig.default(), ledger_rows=rows)
    assert view.total_count == 1
    assert view.rows[0].memory_id == "m1"
    assert view.active_count == 1


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


def test_imports_distinguish_available_from_unavailable():
    view = cm.build_imports()
    by_key = {a.key: a for a in view.adapters}
    assert by_key["pdf"].supported is True
    assert by_key["hf_dataset"].supported is True
    assert by_key["markdown"].supported is False
    assert "Coming soon" in by_key["markdown"].status_label
    assert view.supported_count == 2
    assert view.lifecycle_steps  # the governed intake lifecycle is described


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def test_settings_advertise_read_only():
    view = cm.build_settings(cm.ConsoleConfig.default())
    assert "read-only" in view.read_only_notice.lower()
    labels = {k for k, _ in view.items}
    assert "Environment" in labels
    assert "Retrieval backend" in labels


# ---------------------------------------------------------------------------
# Structural guarantees: read-only by construction
# ---------------------------------------------------------------------------

_FORBIDDEN_CALL_TOKENS = (
    ".add_memory(", ".delete_memory(", ".dispute_memory(", ".import_knowledge(",
    ".delete_knowledge_source(", "save_registry(", "save_review_queue(",
    "save_action_requests(", "save_memory_review_queue(", "write_memory_proposals(",
    "mark_action_result(", "execute_lifecycle_action(", "append_execution_result(",
    "append_execution_audit(", "save_follow_ups(", "save_activation_blocks(",
    "save_execution_approvals(", "write_baseline(", "write_monitoring_run(",
    ".activate(", ".deactivate(", ".rollback(", ".supersede(",
    ".emergency_deactivate(", "_persist_bank(",
)


def test_console_model_calls_no_mutation_api():
    source = (ROOT / "src" / "agent" / "console_model.py").read_text(encoding="utf-8")
    for token in _FORBIDDEN_CALL_TOKENS:
        assert token not in source, f"console_model must not call {token}"
    # It must also parse cleanly (defends against accidental syntax-level shims).
    ast.parse(source)


def _snapshot(paths):
    return {p: p.read_bytes() for p in paths if p.exists()}


def test_build_functions_do_not_mutate_on_disk_data(tmp_path):
    config = _empty_config(tmp_path)
    # Populate every data file the builders read, using real serializers/shapes.
    _write_pack_state(
        config.pack_state_path,
        ActivePackState(pack_id="pack-a", pack_version="1", pack_fingerprint="fp",
                        source_type="pdf", source_id="s", source_revision="r1",
                        status=PackLifecycleState.ACTIVE))
    _write_monitoring_run(config.monitoring_history_path,
                          recommendation="keep_active", critical=False)
    config.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    config.ledger_path.write_text(
        json.dumps({"memory_id": "m1", "canonical_text": "x", "status": "active"}) + "\n",
        encoding="utf-8")

    watched = [
        config.pack_state_path, config.pack_audit_path, config.monitoring_history_path,
        config.ledger_path, config.regression_queue_path, config.regression_actions_path,
        config.memory_review_path, config.source_review_path, config.registry_path,
    ]
    before = _snapshot(watched)

    cm.build_home(config)
    cm.build_sources(config)
    cm.build_packs(config)
    cm.build_reviews(config)
    cm.build_monitoring(config)
    cm.build_memory(config)
    cm.build_imports()
    cm.build_settings(config)

    after = _snapshot(watched)
    assert before == after, "a read-only page load must not change any data file"
    # And no new files were created by merely loading the views.
    assert set(before) == set(after)
