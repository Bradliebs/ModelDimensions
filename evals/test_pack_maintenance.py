"""Tests for the v2.2 source refresh & pack maintenance layer.

These cover the *read-only* freshness analysis in ``agent.pack_maintenance``:
status computation from existing provenance fields, domain-window overrides,
inventory construction over a real library, the refresh plan, the maintenance
report, and the real v2.1 M365 / Coding pack.

Nothing here mutates a source, changes citation eligibility, or touches
concept-cell geometry, grounding, verifier, or lifecycle semantics. A fixed
``now`` keeps every time-based assertion deterministic.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import pack_builder  # noqa: E402
from agent.knowledge_library import KnowledgeLibrary  # noqa: E402
from agent.knowledge_sources import (  # noqa: E402
    KnowledgeChunk,
    KnowledgeDomain,
    KnowledgeSource,
    SourceAuthority,
)
from agent.pack_maintenance import (  # noqa: E402
    FreshnessPolicy,
    FreshnessStatus,
    _assess,
    build_inventory,
    build_maintenance_report,
    build_refresh_plan,
)
from agent.project_packs import PackRegistry  # noqa: E402

_NOW = datetime(2026, 6, 3, tzinfo=timezone.utc)
_MANIFEST = ROOT / "packs" / "m365_coding_assistant" / "pack.yaml"


def _src(staleness_policy: str = "static",
         retrieved_at: str = "",
         domain: KnowledgeDomain = KnowledgeDomain.GENERAL,
         source_id: str = "s1",
         name: str = "Source") -> KnowledgeSource:
    return KnowledgeSource(
        source_id=source_id,
        source_name=name,
        source_type="markdown",
        domain=domain,
        authority=SourceAuthority.REPUTABLE,
        path_or_url="local.md",
        staleness_policy=staleness_policy,
        retrieved_at=retrieved_at,
    )


def _add_chunk(lib: KnowledgeLibrary, source_id: str, text: str) -> None:
    """Insert a chunk directly (the library has no public add_chunk)."""
    src = lib.get_source(source_id)
    chunk_id = f"chk-{source_id}-{len(lib._chunks)}"
    lib._chunks[chunk_id] = KnowledgeChunk(
        chunk_id=chunk_id,
        source_id=source_id,
        document_id=f"doc-{source_id}",
        chunk_text=text,
        domain=src.domain,
        authority=src.authority,
        source_name=src.source_name,
        source_section="",
    )


# 1. declared policies short-circuit the time-based check. ------------------

def test_static_policy_is_current():
    status, _reason, age = _assess(_src("static"), FreshnessPolicy(), _NOW)
    assert status is FreshnessStatus.CURRENT
    assert age is None


def test_stale_policy_is_stale():
    status, _reason, _age = _assess(_src("stale"), FreshnessPolicy(), _NOW)
    assert status is FreshnessStatus.STALE


def test_review_required_policy_is_review_due():
    status, _reason, _age = _assess(
        _src("review_required"), FreshnessPolicy(), _NOW)
    assert status is FreshnessStatus.REVIEW_DUE


# 2. time-based windows for non-declared policies. --------------------------

def test_old_date_is_stale():
    # 400 days before _NOW, default window 365 -> stale.
    status, _reason, age = _assess(
        _src("dated", retrieved_at="2025-04-29"), FreshnessPolicy(), _NOW)
    assert status is FreshnessStatus.STALE
    assert age is not None and age >= 365


def test_mid_age_is_review_due():
    # ~300 days old: past 0.8*365=292 review threshold, below 365 window.
    status, _reason, _age = _assess(
        _src("dated", retrieved_at="2025-08-07"), FreshnessPolicy(), _NOW)
    assert status is FreshnessStatus.REVIEW_DUE


def test_fresh_date_is_current():
    status, _reason, _age = _assess(
        _src("dated", retrieved_at="2026-05-01"), FreshnessPolicy(), _NOW)
    assert status is FreshnessStatus.CURRENT


def test_no_date_with_time_policy_is_unknown():
    status, _reason, age = _assess(
        _src("dated", retrieved_at=""), FreshnessPolicy(), _NOW)
    assert status is FreshnessStatus.UNKNOWN
    assert age is None


# 3. domain window override. ------------------------------------------------

def test_domain_window_override_makes_source_stale():
    policy = FreshnessPolicy(
        default_stale_after_days=365,
        domain_stale_after_days={"microsoft": 180})
    # 200 days old: current under the 365 default, stale under the 180 override.
    src = _src("dated", retrieved_at="2025-11-15",
               domain=KnowledgeDomain.MICROSOFT)
    status, _reason, _age = _assess(src, policy, _NOW)
    assert status is FreshnessStatus.STALE


# 4. inventory covers every active source and counts its chunks. ------------

def test_build_inventory_counts_chunks_and_covers_sources():
    lib = KnowledgeLibrary()
    lib.add_source(_src("static", source_id="a", name="Alpha"))
    lib.add_source(_src("stale", source_id="b", name="Bravo"))
    _add_chunk(lib, "a", "alpha one")
    _add_chunk(lib, "a", "alpha two")
    _add_chunk(lib, "b", "bravo one")
    inv = build_inventory(lib, FreshnessPolicy(), now=_NOW)
    by_id = {r.source_id: r for r in inv}
    assert set(by_id) == {"a", "b"}
    assert by_id["a"].entries == 2
    assert by_id["b"].entries == 1
    # stale sorts before current.
    assert inv[0].status is FreshnessStatus.STALE


# 5. refresh plan excludes current and sorts highest-risk first. ------------

def test_refresh_plan_excludes_current_and_ranks_risk():
    lib = KnowledgeLibrary()
    lib.add_source(_src("static", source_id="a", name="Alpha"))
    lib.add_source(_src("review_required", source_id="b", name="Bravo"))
    lib.add_source(_src("stale", source_id="c", name="Charlie"))
    inv = build_inventory(lib, FreshnessPolicy(), now=_NOW)
    plan = build_refresh_plan(inv)
    ids = [i.source_id for i in plan]
    assert "a" not in ids                # current excluded
    assert plan[0].source_id == "c"      # high-risk stale first
    assert plan[0].risk == "high"


# 6. maintenance report tallies statuses. -----------------------------------

def test_maintenance_report_counts():
    lib = KnowledgeLibrary()
    lib.add_source(_src("static", source_id="a", name="Alpha"))
    lib.add_source(_src("review_required", source_id="b", name="Bravo"))
    lib.add_source(_src("stale", source_id="c", name="Charlie"))
    _add_chunk(lib, "a", "x")
    inv = build_inventory(lib, FreshnessPolicy(), now=_NOW)
    report = build_maintenance_report(inv, eval_total=20, eval_passed=20)
    assert report.source_count == 3
    assert report.current_count == 1
    assert report.review_due_count == 1
    assert report.stale_count == 1
    assert report.unknown_count == 0
    assert report.total_entries == 1
    assert report.eval_pass_rate == 1.0


def test_policy_from_manifest_reads_freshness_block():
    manifest = {
        "policy": {
            "freshness": {
                "default_stale_after_days": 200,
                "domain_stale_after_days": {"microsoft": 90},
                "review_due_fraction": 0.5,
            }
        }
    }
    policy = FreshnessPolicy.from_manifest(manifest)
    assert policy.default_stale_after_days == 200
    assert policy.domain_stale_after_days["microsoft"] == 90
    assert policy.review_due_fraction == 0.5


def test_policy_from_manifest_defaults_when_absent():
    policy = FreshnessPolicy.from_manifest({})
    assert policy.default_stale_after_days == 365
    assert policy.review_due_fraction == 0.8


# 7. the real v2.1 M365 / Coding pack. --------------------------------------

def test_m365_pack_maintenance(tmp_path, monkeypatch):
    """Build the shipped pack and assert its declared-policy freshness view.

    Every source short-circuits on its declared ``staleness_policy``: the Power
    Platform source is ``stale``; the other four are ``review_required``. So
    domain windows do not change the live result, which makes the maintenance
    view fully deterministic.
    """
    monkeypatch.chdir(ROOT)
    registry = PackRegistry(tmp_path / "packs")
    plan = pack_builder.PackBuildPlan.from_file(_MANIFEST)
    report = pack_builder.build_pack(plan, registry)
    service = pack_builder.open_pack_service(registry, report.pack_id)

    import yaml

    raw = yaml.safe_load(_MANIFEST.read_text(encoding="utf-8"))
    policy = FreshnessPolicy.from_manifest(raw)
    inv = build_inventory(service.knowledge, policy, now=_NOW)
    maint = build_maintenance_report(inv)

    assert maint.source_count == 5
    assert maint.stale_count == 1
    assert maint.review_due_count == 4
    assert maint.current_count == 0
    assert maint.total_entries > 0

    stale = [r for r in inv if r.status is FreshnessStatus.STALE]
    assert len(stale) == 1
    assert stale[0].source_name == "Power Platform PowerApps Formulas"

    plan_items = build_refresh_plan(inv)
    assert len(plan_items) == 5            # every source needs attention
    assert plan_items[0].risk == "high"    # the stale one ranks first
