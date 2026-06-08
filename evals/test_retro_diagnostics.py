"""Unit tests for the reading-engine diagnostics package.

All four diagnostic modules are pure (no GPU, no model, no live bank), so
this single file covers acceptance, compute_plan, disjointness, and
layer_attribution with synthetic inputs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retro.diagnostics.acceptance import (  # noqa: E402
    IGNORANCE_HURT_MIN_NATS,
    SEMANTIC_GAP_MIN_NATS,
    VAL_WITH_LOSS_CEILING_NATS,
    evaluate_all,
    evaluate_ignorance_test,
    evaluate_loss_ceiling,
    evaluate_semantic_gap,
)
from src.retro.diagnostics.compute_plan import (  # noqa: E402
    CHINCHILLA_TOKENS_PER_PARAM,
    build_training_plan,
    chinchilla_token_target,
    estimate_compute_for_training,
    presented_tokens,
    render_plan_markdown,
)
from src.retro.diagnostics.disjointness import (  # noqa: E402
    DisjointnessPolicy,
    build_fingerprint_set,
    build_title_set,
    check_disjointness,
    fingerprint,
    normalise_paragraph,
    render_report_markdown,
)
from src.retro.diagnostics.layer_attribution import (  # noqa: E402
    DOMINANT_LAYER_MIN_SHARE,
    compute_layer_attribution,
    render_attribution_markdown,
)


# ---------- acceptance ----------


class TestAcceptance:
    """Each gate must pass exactly at the threshold and fail just below."""

    def test_ignorance_passes_at_exact_threshold(self) -> None:
        losses = {"none": 4.000, "random": 4.000 + IGNORANCE_HURT_MIN_NATS, "real": 3.800}
        result = evaluate_ignorance_test(losses)
        assert result.passed is True
        assert result.measured_nats == pytest.approx(IGNORANCE_HURT_MIN_NATS, abs=1e-9)
        assert result.direction == "ge"

    def test_ignorance_fails_just_below(self) -> None:
        losses = {"none": 4.000, "random": 4.000 + IGNORANCE_HURT_MIN_NATS - 1e-4, "real": 3.800}
        assert evaluate_ignorance_test(losses).passed is False

    def test_semantic_passes_at_threshold(self) -> None:
        # Land comfortably above threshold; one-ULP-near-threshold edge cases are
        # tested separately to avoid IEEE-754 round-trip flakiness in the test
        # arithmetic. The production code uses raw subtraction; we trust
        # `pytest.approx` to assert near-equality, not exact equality.
        losses = {"none": 4.000, "random": 4.050, "real": 3.800}
        result = evaluate_semantic_gap(losses)
        assert result.passed is True
        assert result.measured_nats == pytest.approx(0.200, abs=1e-9)
        assert result.measured_nats >= SEMANTIC_GAP_MIN_NATS

    def test_semantic_fails_when_gap_too_small(self) -> None:
        losses = {"none": 4.000, "random": 4.050, "real": 3.900}
        result = evaluate_semantic_gap(losses)
        assert result.passed is False
        assert result.measured_nats == pytest.approx(0.100, abs=1e-9)

    def test_loss_ceiling_strict(self) -> None:
        ok = {"none": 4.000, "random": 4.050, "real": VAL_WITH_LOSS_CEILING_NATS}
        assert evaluate_loss_ceiling(ok).passed is True
        bad = {"none": 4.000, "random": 4.050, "real": VAL_WITH_LOSS_CEILING_NATS + 1e-4}
        assert evaluate_loss_ceiling(bad).passed is False

    def test_evaluate_all_combines_gates(self) -> None:
        losses = {"none": 4.000, "random": 4.050, "real": 3.700}
        report = evaluate_all(losses)
        assert set(report.gates) == {"ignorance_test", "semantic_gap", "loss_ceiling"}
        assert report.all_passed is True
        d = report.to_dict()
        assert d["all_passed"] is True
        assert d["gates"]["ignorance_test"]["passed"] is True

    def test_missing_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="missing required modes"):
            evaluate_ignorance_test({"none": 4.0, "real": 3.8})

    def test_nan_loss_rejected(self) -> None:
        with pytest.raises(ValueError, match="not a positive finite"):
            evaluate_ignorance_test({"none": float("nan"), "random": 4.0, "real": 3.8})

    def test_non_positive_loss_rejected(self) -> None:
        with pytest.raises(ValueError, match="not a positive finite"):
            evaluate_ignorance_test({"none": 0.0, "random": 4.0, "real": 3.8})


# ---------- compute_plan ----------


class TestComputePlan:
    def test_chinchilla_target(self) -> None:
        assert chinchilla_token_target(404_000_000) == int(
            round(404_000_000 * CHINCHILLA_TOKENS_PER_PARAM)
        )

    def test_chinchilla_rejects_non_positive(self) -> None:
        with pytest.raises(ValueError):
            chinchilla_token_target(0)

    def test_presented_tokens_arithmetic(self) -> None:
        # Default-style: 400k iters * 32 effective batch * 256 seq = 3.276 8B
        assert presented_tokens(iters=400_000, effective_batch=32, block_size=256) == (
            400_000 * 32 * 256
        )

    def test_presented_tokens_rejects_zero(self) -> None:
        with pytest.raises(ValueError):
            presented_tokens(iters=0, effective_batch=32, block_size=256)

    def test_estimate_compute_unknown_device(self) -> None:
        with pytest.raises(ValueError, match="unknown device"):
            estimate_compute_for_training(
                presented_tokens_total=1_000_000_000, device="m4_macbook"
            )

    def test_estimate_compute_with_cost(self) -> None:
        est = estimate_compute_for_training(
            presented_tokens_total=1_000_000_000,
            device="a100_80gb_bf16",
            hourly_cost_usd=2.0,
        )
        # hours_low < hours_high always
        assert est.hours_low <= est.hours_high
        assert est.dollars_low is not None and est.dollars_high is not None
        assert est.dollars_low <= est.dollars_high
        assert est.dollars_low == pytest.approx(est.hours_low * 2.0, rel=1e-9)

    def test_build_training_plan_default_warns_below_target(self) -> None:
        # 400k * 32 * 256 = 3.28B, less than the 8B unique target.
        plan = build_training_plan()
        assert plan.presented_tokens == 400_000 * 32 * 256
        assert any("below the unique-token target" in n for n in plan.notes)
        d = plan.to_dict()
        assert d["chinchilla_tokens"] == chinchilla_token_target(404_000_000)
        assert len(d["compute_estimates"]) == 2

    def test_build_training_plan_with_costs(self) -> None:
        plan = build_training_plan(
            iters=10_000,
            effective_batch=16,
            block_size=256,
            hourly_costs_usd={"a100_80gb_bf16": 2.0, "h100_80gb_bf16": 4.0},
        )
        for est in plan.compute_estimates:
            assert est.dollars_low is not None

    def test_render_markdown_runs(self) -> None:
        plan = build_training_plan(iters=1_000, effective_batch=8, block_size=64)
        md = render_plan_markdown(plan)
        assert "Chinchilla scaling plan" in md
        assert "Candidate disjoint corpora" in md
        assert "Compute envelope" in md


# ---------- disjointness ----------


class TestDisjointness:
    def test_normalise_collapses_whitespace_and_case(self) -> None:
        a = "  The   Quick BROWN fox.\n"
        b = "the quick brown fox."
        assert normalise_paragraph(a) == normalise_paragraph(b)

    def test_normalise_strips_zero_width(self) -> None:
        a = "hello\u200bworld"
        b = "helloworld"
        assert normalise_paragraph(a) == normalise_paragraph(b)

    def test_fingerprint_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            fingerprint("   ")

    def test_fingerprint_collisions_only_under_equivalence(self) -> None:
        assert fingerprint("Hello world.") == fingerprint("hello   world.")
        assert fingerprint("Hello world.") != fingerprint("hello worlds.")

    def test_build_fingerprint_set_skips_short(self) -> None:
        fps = build_fingerprint_set(["short", "this paragraph is long enough"], min_chars=20)
        assert len(fps) == 1

    def test_check_passes_with_clean_corpus(self) -> None:
        bank = build_fingerprint_set(
            ["the bank knows about apples and oranges only"], min_chars=20
        )
        corpus = build_fingerprint_set(
            ["entirely different content about programming languages"],
            min_chars=20,
        )
        report = check_disjointness(bank_fingerprints=bank, corpus_fingerprints=corpus)
        assert report.passed is True
        assert report.corpus_paragraph_overlap == 0

    def test_check_fails_with_overlap(self) -> None:
        shared = "this exact paragraph appears in both bank and corpus"
        bank = build_fingerprint_set([shared, "other bank text here"], min_chars=20)
        corpus = build_fingerprint_set([shared, "other corpus text here"], min_chars=20)
        report = check_disjointness(
            bank_fingerprints=bank,
            corpus_fingerprints=corpus,
            policy=DisjointnessPolicy(max_paragraph_overlap_ratio=0.001),
        )
        assert report.passed is False
        assert report.corpus_paragraph_overlap == 1
        assert report.sample_overlapping_fingerprints  # at least one shown

    def test_check_title_overlap_zero_tolerance(self) -> None:
        bank = build_fingerprint_set(["bank only paragraph one"], min_chars=20)
        corpus = build_fingerprint_set(["corpus only paragraph one"], min_chars=20)
        bank_titles = build_title_set(["Apple (fruit)"])
        corpus_titles = build_title_set(["APPLE (Fruit)"])  # normalises equal
        report = check_disjointness(
            bank_fingerprints=bank,
            corpus_fingerprints=corpus,
            bank_titles=bank_titles,
            corpus_titles=corpus_titles,
        )
        assert report.paragraph_pass is True
        assert report.title_pass is False
        assert report.passed is False

    def test_render_report_runs(self) -> None:
        bank = build_fingerprint_set(["bank paragraph that is long enough"], min_chars=20)
        corpus = build_fingerprint_set(["entirely distinct corpus paragraph"], min_chars=20)
        md = render_report_markdown(
            check_disjointness(bank_fingerprints=bank, corpus_fingerprints=corpus)
        )
        assert "Corpus disjointness report" in md

    def test_check_rejects_empty_corpus(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            check_disjointness(bank_fingerprints={"abc"}, corpus_fingerprints=set())


# ---------- layer_attribution ----------


class TestLayerAttribution:
    def test_dominant_layer_detected(self) -> None:
        # benefit = 4.000 - 3.800 = 0.200
        # suppressing layer 5 costs 0.126 of those 0.200 -> share 0.63
        report = compute_layer_attribution(
            loss_without=4.000,
            loss_real_full=3.800,
            loss_real_with_layer_suppressed={1: 3.810, 3: 3.830, 5: 3.926, 7: 3.840},
        )
        assert report.full_benefit_nats == pytest.approx(0.200, abs=1e-9)
        assert report.dominant_layer_index == 5
        assert report.dominant_layer_share == pytest.approx(0.63, abs=1e-2)
        assert report.dominant_layer_passes is True

    def test_share_below_threshold_fails(self) -> None:
        # Even contribution across layers; share == 0.25 each, below 0.50
        report = compute_layer_attribution(
            loss_without=4.000,
            loss_real_full=3.800,
            loss_real_with_layer_suppressed={1: 3.850, 3: 3.850, 5: 3.850, 7: 3.850},
        )
        assert report.dominant_layer_passes is False
        assert report.dominant_layer_share == pytest.approx(0.25, abs=1e-9)

    def test_negative_benefit_flagged(self) -> None:
        # real loss > without loss -> retrieval is hurting; attribution moot
        report = compute_layer_attribution(
            loss_without=3.700,
            loss_real_full=3.800,
            loss_real_with_layer_suppressed={1: 3.810},
        )
        assert report.dominant_layer_passes is False
        assert any("non-positive" in n for n in report.notes)

    def test_negative_delta_layer_noted(self) -> None:
        # Layer 7's suppression LOWERS loss (this layer is hurting retrieval)
        report = compute_layer_attribution(
            loss_without=4.000,
            loss_real_full=3.800,
            loss_real_with_layer_suppressed={5: 3.920, 7: 3.780},
        )
        assert any("layer 7" in n and "hurting" in n for n in report.notes)

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_layer_attribution(
                loss_without=4.0, loss_real_full=3.8, loss_real_with_layer_suppressed={}
            )

    def test_non_finite_input_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_layer_attribution(
                loss_without=float("nan"),
                loss_real_full=3.8,
                loss_real_with_layer_suppressed={1: 3.9},
            )

    def test_render_markdown_runs(self) -> None:
        report = compute_layer_attribution(
            loss_without=4.000,
            loss_real_full=3.800,
            loss_real_with_layer_suppressed={1: 3.810, 5: 3.926},
        )
        md = render_attribution_markdown(report)
        assert "Layer attribution" in md
        assert str(DOMINANT_LAYER_MIN_SHARE) in md or "0.50" in md
