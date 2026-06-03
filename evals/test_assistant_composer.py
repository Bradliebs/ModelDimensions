"""Tests for the v2.0 optional SLM-grounded assistant composer layer.

The composer is a *rendering* layer on top of the frozen retrieve -> verify ->
ground path. It never decides grounding: the deterministic
:class:`GroundingPackage` fixes the mode, the refused flag, and the exact set of
citable ids. These tests prove the safety contract holds whether the prose is
written by the deterministic template or by an optional local SLM (mocked here
so the suite stays offline): an SLM cannot invent a citation, cannot ground a
refused answer, and malformed SLM output falls back to the template. They also
confirm the structural guarantees of the underlying path survive into the
composed answer -- deleted and superseded memories are never citable, and the
medical informational-only flag is carried through.

Nothing here touches concept-cell geometry, grounding policy, lifecycle
semantics, or the knowledge source/authority model; every service is built into
an isolated ``tmp_path`` so the suite leaves no trace.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent.memory_lifecycle import LifecycleVerdict  # noqa: E402,F401
from agent.workbench_service import WorkbenchService  # noqa: E402
from slm.assistant_composer import (  # noqa: E402
    ComposerMode,
    LocalSLMComposer,
    TemplateComposer,
)
from slm.local_slm_backend import MockSLMBackend  # noqa: E402

_FRIDAY = "the supplier delivery is on Friday afternoon"
_MONDAY = "the supplier delivery is on Monday afternoon"
_CODING_DOC = (
    "# Pydantic\n\n"
    "Pydantic validates a typed model on construction and raises on bad input.\n"
)
_MEDICAL_DOC = (
    "# Influenza\n\n"
    "Common flu symptoms include fever, cough, and fatigue lasting several "
    "days.\n"
)


def _service(tmp_path) -> WorkbenchService:
    return WorkbenchService(
        ledger_path=str(tmp_path / "ledger.jsonl"),
        queue_path=str(tmp_path / "queue.jsonl"),
        knowledge_path=str(tmp_path / "knowledge.jsonl"),
    )


def _write(tmp_path, text: str, name: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _write_note(tmp_path, text: str, name: str = "note.md") -> Path:
    return _write(tmp_path, text, name)


def _proposal_for(service, text: str):
    target = text.lower().rstrip(".")
    for p in service.list_proposals(status=None):
        if p.canonical_text.lower().rstrip(".") == target:
            return p
    raise AssertionError(f"no proposal matched: {text!r}")


# 1. the template composer grounds an accepted memory. ----------------------

def test_template_composer_grounds_accepted_memory(tmp_path):
    service = _service(tmp_path)
    entry = service.add_memory(_FRIDAY, source="seed")

    package = service.build_grounding_package(_FRIDAY)
    answer = TemplateComposer().compose(package)

    assert package.mode == ComposerMode.GROUNDED
    assert answer.refused is False
    assert answer.fell_back is False
    assert answer.citations == [f"mem:{entry.memory_id}"]
    assert f"mem:{entry.memory_id}" in answer.text


# 2. no evidence yields a refusal. ------------------------------------------

def test_no_evidence_yields_refusal(tmp_path):
    service = _service(tmp_path)

    package = service.build_grounding_package("the zebra quantum teapot orbits")
    answer = TemplateComposer().compose(package)

    assert package.mode == ComposerMode.REFUSAL
    assert answer.refused is True
    assert answer.citations == []


# 3. an SLM cannot invent a citation. ---------------------------------------

def test_slm_cannot_invent_citation(tmp_path):
    service = _service(tmp_path)
    service.add_memory(_FRIDAY, source="seed")
    package = service.build_grounding_package(_FRIDAY)

    backend = MockSLMBackend("The delivery is Friday [mem:fabricated-id].")
    answer = LocalSLMComposer(backend).compose(package)

    # The invented citation is rejected, so the composer falls back.
    assert answer.fell_back is True
    assert "fabricated-id" not in answer.text
    assert answer.composer_backend == "slm:mock-slm"


# 4. an SLM cannot turn a refused answer into a grounded one. ---------------

def test_slm_cannot_ground_refused_answer(tmp_path):
    service = _service(tmp_path)
    package = service.build_grounding_package("the zebra quantum teapot orbits")

    backend = MockSLMBackend("Yes, the answer is certain [mem:made-up].")
    answer = LocalSLMComposer(backend).compose(package)

    # A citation on a non-grounded mode is rejected -> safe template refusal.
    assert answer.refused is True
    assert answer.mode == ComposerMode.REFUSAL
    assert answer.fell_back is True
    assert answer.citations == []


# 5. malformed SLM output falls back to the template. -----------------------

def test_malformed_slm_output_falls_back(tmp_path):
    service = _service(tmp_path)
    service.add_memory(_FRIDAY, source="seed")
    package = service.build_grounding_package(_FRIDAY)

    template_text = TemplateComposer().compose(package).text
    answer = LocalSLMComposer(MockSLMBackend("")).compose(package)

    assert answer.fell_back is True
    assert answer.text == template_text


# 6. a deleted memory cannot be cited. --------------------------------------

def test_deleted_memory_cannot_be_cited(tmp_path):
    service = _service(tmp_path)
    entry = service.add_memory(_FRIDAY, source="seed")

    # Grounded before deletion.
    assert service.build_grounding_package(_FRIDAY).mode == ComposerMode.GROUNDED

    service.delete_memory(entry.memory_id)
    package = service.build_grounding_package(_FRIDAY)
    answer = TemplateComposer().compose(package)

    assert answer.refused is True
    assert package.evidence == []
    assert f"mem:{entry.memory_id}" not in answer.text


# 7. a superseded memory cannot be cited as the current answer. -------------

def test_superseded_memory_cannot_be_cited_as_current(tmp_path):
    service = _service(tmp_path)
    old = service.add_memory(_FRIDAY, source="seed")
    note = _write_note(tmp_path, f"# Facts\n\n- {_MONDAY}\n")
    service.import_notes(note)
    proposal = _proposal_for(service, _MONDAY)
    service.approve_proposal_superseding(proposal.proposal_id, old.memory_id)
    service.write_approved_proposals()

    package = service.build_grounding_package(_FRIDAY)
    answer = TemplateComposer().compose(package)

    # The superseded Friday memory is gone from the bank: not grounded, never
    # cited, and surfaced only as a display-only historical note.
    assert answer.refused is True
    assert f"mem:{old.memory_id}" not in [e.citation_id for e in package.evidence]
    assert package.historical_note is not None
    assert old.memory_id in package.historical_note


# 8. a medical knowledge answer carries the informational-only flag. --------

def test_medical_answer_is_informational_only(tmp_path):
    service = _service(tmp_path)
    doc = _write(tmp_path, _MEDICAL_DOC, "flu.md")
    service.import_knowledge(doc, domain="medical", authority="reputable",
                             source_name="Flu Notes", version="v1")

    package = service.build_grounding_package("what are the flu symptoms")
    answer = TemplateComposer().compose(package)

    assert package.mode == ComposerMode.GROUNDED
    assert package.informational_only is True
    assert answer.informational_only is True
    assert "Informational only" in answer.text


# 9. a model-prior answer is labelled when explicitly allowed. --------------

def test_model_prior_output_is_labelled(tmp_path):
    service = _service(tmp_path)

    package = service.build_grounding_package(
        "the zebra quantum teapot orbits", allow_model_prior=True)
    answer = TemplateComposer().compose(package)

    assert package.mode == ComposerMode.MODEL_PRIOR_LABELLED
    assert answer.model_prior_labelled is True
    assert "[Unverified model prior]" in answer.text
    assert answer.citations == []


# 10. the composer backend name appears in the assistant audit. -------------

def test_composer_backend_name_appears_in_audit(tmp_path):
    service = _service(tmp_path)
    service.add_memory(_FRIDAY, source="seed")

    template_result = service.answer_query(_FRIDAY)
    assert template_result.composer_backend == "template"
    assert template_result.to_dict()["composer_backend"] == "template"

    slm_result = service.answer_query(
        _FRIDAY, use_slm=True, slm_backend=MockSLMBackend("Delivery is Friday."))
    assert slm_result.composer_backend == "slm:mock-slm"
    assert slm_result.to_dict()["composer_backend"] == "slm:mock-slm"
