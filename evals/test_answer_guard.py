"""Tests for the AnswerGuard (the assistant-layer immune system).

The guard is an independent, deterministic post-check over a composed answer.
These tests cover three things:

* **Invariants** — the built-in template composer always passes the guard, on
  every mode (grounded, refusal, conflict, model-prior, pack-summary).
* **Detection** — a deliberately misbehaving custom composer (one that invents
  a citation, softens a refusal, drops a stale caution, or relabels the mode)
  is caught with the right violation code.
* **Enforcement** — ``enforce`` discards a bad answer and recomposes a clean
  template answer, while still reporting the original rejection; and the real
  ``answer_query`` path records an ``ACCEPT`` verdict in the audit.

Nothing here touches frozen grounding, verifier, lifecycle, or citation policy.
The guard only *observes* the composed answer; it never relaxes a rule.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from slm.answer_guard import (  # noqa: E402
    CITATION_ON_NONGROUNDED,
    DROPPED_STALE_CAUTION,
    INVENTED_CITATION,
    MODE_MISMATCH,
    SOFTENED_REFUSAL,
    UNLABELLED_MODEL_PRIOR,
    check_answer,
    enforce,
)
from slm.assistant_composer import (  # noqa: E402
    AssistantComposer,
    ComposedAnswer,
    ComposerMode,
    EvidenceItem,
    GroundingPackage,
    TemplateComposer,
)


def _grounded_package(*, stale: bool = False) -> GroundingPackage:
    cautions = (["Source 'X' is marked stale (staleness policy: stale); it "
                 "may be outdated — verify."] if stale else [])
    return GroundingPackage(
        query="q",
        mode=ComposerMode.GROUNDED,
        memory_used=True,
        knowledge_used=False,
        model_prior_used=False,
        informational_only=False,
        refused=False,
        route="memory_only",
        evidence=[EvidenceItem(
            citation_id="mem:m-1", kind="memory", text="a grounded fact")],
        cautions=cautions,
    )


def _refusal_package() -> GroundingPackage:
    return GroundingPackage(
        query="q",
        mode=ComposerMode.REFUSAL,
        memory_used=False,
        knowledge_used=False,
        model_prior_used=False,
        informational_only=False,
        refused=True,
        route="general_model_not_grounded",
    )


def _model_prior_package() -> GroundingPackage:
    return GroundingPackage(
        query="q",
        mode=ComposerMode.MODEL_PRIOR_LABELLED,
        memory_used=False,
        knowledge_used=False,
        model_prior_used=True,
        informational_only=False,
        refused=False,
        route="general_model_not_grounded",
    )


# 1. the template composer passes the guard on every mode. ------------------

def test_template_always_passes_guard():
    composer = TemplateComposer()
    for package in (_grounded_package(), _grounded_package(stale=True),
                    _refusal_package(), _model_prior_package()):
        answer = composer.compose(package)
        report = check_answer(package, answer)
        assert report.ok, f"{package.mode}: {report.to_dict()}"


# 2. an invented citation is caught. ----------------------------------------

def test_invented_citation_caught():
    package = _grounded_package()
    answer = ComposedAnswer(
        text="Based on evidence [mem:m-1] and also [mem:GHOST].",
        mode=ComposerMode.GROUNDED,
        citations=["mem:m-1", "mem:GHOST"],
        composer_backend="rogue",
        model_prior_labelled=False,
        informational_only=False,
        refused=False,
    )
    report = check_answer(package, answer)
    assert not report.ok
    assert any(v.code == INVENTED_CITATION for v in report.violations)


# 3. a softened refusal is caught. ------------------------------------------

def test_softened_refusal_caught():
    package = _refusal_package()
    answer = ComposedAnswer(
        text="Sure, here is a confident answer anyway.",
        mode=ComposerMode.REFUSAL,
        citations=[],
        composer_backend="rogue",
        model_prior_labelled=False,
        informational_only=False,
        refused=False,  # softened: package refused, answer did not
    )
    report = check_answer(package, answer)
    assert not report.ok
    assert any(v.code == SOFTENED_REFUSAL for v in report.violations)


# 4. an unlabelled model prior is caught. -----------------------------------

def test_unlabelled_model_prior_caught():
    package = _model_prior_package()
    answer = ComposedAnswer(
        text="The answer is definitely 42.",
        mode=ComposerMode.MODEL_PRIOR_LABELLED,
        citations=[],
        composer_backend="rogue",
        model_prior_labelled=False,  # not labelled
        informational_only=False,
        refused=False,
    )
    report = check_answer(package, answer)
    assert not report.ok
    assert any(v.code == UNLABELLED_MODEL_PRIOR for v in report.violations)


# 5. citing a source on a refusal is caught. --------------------------------

def test_citation_on_nongrounded_caught():
    package = _refusal_package()
    answer = ComposedAnswer(
        text="No grounded source, but see [src:made-up].",
        mode=ComposerMode.REFUSAL,
        citations=[],
        composer_backend="rogue",
        model_prior_labelled=False,
        informational_only=False,
        refused=True,
    )
    report = check_answer(package, answer)
    assert not report.ok
    codes = {v.code for v in report.violations}
    assert CITATION_ON_NONGROUNDED in codes
    assert INVENTED_CITATION in codes  # src:made-up is also not allowed


# 6. dropping a stale caution is caught. ------------------------------------

def test_dropped_stale_caution_caught():
    package = _grounded_package(stale=True)
    answer = ComposedAnswer(
        text="Based on evidence [mem:m-1]. (caution silently omitted)",
        mode=ComposerMode.GROUNDED,
        citations=["mem:m-1"],
        composer_backend="rogue",
        model_prior_labelled=False,
        informational_only=False,
        refused=False,
    )
    report = check_answer(package, answer)
    assert not report.ok
    assert any(v.code == DROPPED_STALE_CAUTION for v in report.violations)


# 7. a relabelled mode is caught. -------------------------------------------

def test_mode_mismatch_caught():
    package = _refusal_package()
    answer = ComposedAnswer(
        text="Pretending this is grounded.",
        mode=ComposerMode.GROUNDED,  # relabelled
        citations=[],
        composer_backend="rogue",
        model_prior_labelled=False,
        informational_only=False,
        refused=True,
    )
    report = check_answer(package, answer)
    assert not report.ok
    assert any(v.code == MODE_MISMATCH for v in report.violations)


# 8. enforce discards a bad answer and recomposes safely. -------------------

def test_enforce_recomposes_on_reject():
    package = _refusal_package()
    bad = ComposedAnswer(
        text="Confident wrong answer.",
        mode=ComposerMode.GROUNDED,
        citations=["mem:GHOST"],
        composer_backend="rogue",
        model_prior_labelled=False,
        informational_only=False,
        refused=False,
    )
    safe, report = enforce(package, bad)
    assert not report.ok  # the original answer was rejected
    # the replacement is the clean deterministic template answer
    assert safe.refused is True
    assert safe.mode == ComposerMode.REFUSAL
    assert safe.composer_backend == "template"
    # and the safe answer itself passes the guard
    assert check_answer(package, safe).ok


# 9. a rogue custom composer is neutralised through answer_query. -----------

class _RogueComposer(AssistantComposer):
    """A composer that ignores the package and invents a grounded answer."""

    name = "rogue"

    def compose(self, package: GroundingPackage) -> ComposedAnswer:
        return ComposedAnswer(
            text="Trust me, the answer is [mem:GHOST].",
            mode=ComposerMode.GROUNDED,
            citations=["mem:GHOST"],
            composer_backend=self.name,
            model_prior_labelled=False,
            informational_only=False,
            refused=False,
        )


def _service(tmp_path, monkeypatch):
    from agent import pack_builder
    from agent.project_packs import PackRegistry
    monkeypatch.chdir(ROOT)
    manifest = ROOT / "packs" / "m365_coding_assistant" / "pack.yaml"
    registry = PackRegistry(tmp_path / "packs")
    plan = pack_builder.PackBuildPlan.from_file(manifest)
    report = pack_builder.build_pack(plan, registry)
    return pack_builder.open_pack_service(registry, report.pack_id)


def test_answer_query_records_guard_verdict(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    # A verbatim grounded query through the default template composer: the
    # guard must ACCEPT and the verdict must be in the audit trail.
    result = service.answer_query(
        "What did we decide about the production database rollout plan?")
    assert "guard" in result.audit
    assert result.audit["guard"]["verdict"] == "ACCEPT"


def test_answer_query_neutralises_rogue_composer(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    # An empty-ledger decision-recall refuses; a rogue composer tries to turn
    # it into a fabricated grounded answer. enforce must neutralise it.
    result = service.answer_query(
        "What did we decide about the production database rollout plan?",
        composer=_RogueComposer())
    assert result.audit["guard"]["verdict"] == "REJECT"
    # the delivered answer is the safe template refusal, not the fabrication
    assert "GHOST" not in result.answer.text
    assert result.refused is True
