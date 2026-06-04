"""Tests for the v6.3 Hugging Face Metadata Adapter (metadata inspection only).

The adapter converts Hugging Face-style dataset card metadata into an
``ExternalDatasetCandidate`` and runs it through the existing v6.2 Clean Data
Intake Lane. These tests pin its contract:

* deterministic metadata mapping and conservative risk inference;
* gated, private, missing-licence, and possible-PII datasets never auto-classify
  as ``approved_for_knowledge``;
* ``approved_for_eval`` stays distinct from ``approved_for_knowledge``;
* the adapter downloads no dataset rows, imports no durable-state writer, writes
  nothing in stdout mode, and writes only the assessment report under ``--out``.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import hf_data_adapter as hfa  # noqa: E402
from agent.data_intake import (  # noqa: E402
    DataIntakeDecision,
    ExternalDatasetCandidate,
    FindingCode,
    IntakeLane,
)
from agent.hf_data_adapter import (  # noqa: E402
    HuggingFaceAdapterResult,
    HuggingFaceDatasetMetadata,
    assess_hf_metadata,
    hf_metadata_to_candidate,
    load_hf_metadata,
)

_DEMO = ROOT / "demos" / "hf_dataset_metadata_examples.jsonl"
_README = ROOT / "README.md"
_NOW = datetime(2026, 6, 4, tzinfo=timezone.utc)


def _meta(**overrides) -> HuggingFaceDatasetMetadata:
    """A clean, open-licensed, non-gated HF metadata record."""
    base = dict(
        dataset_id="demo/open-qa",
        license="apache-2.0",
        language=("en",),
        task_categories=("question-answering",),
        size_categories=("10K<n<100K",),
        pretty_name="Open QA",
        description="Open-licensed question/answer pairs from public-domain text.",
        homepage="https://huggingface.co/datasets/demo/open-qa",
        citation="@misc{openqa}",
        last_modified="2025-11-01",
        gated=False,
        private=False,
        contains_personal_data=False,
    )
    base.update(overrides)
    return HuggingFaceDatasetMetadata(**base)


# 1. maps open licence metadata to ExternalDatasetCandidate. -----------------

def test_maps_open_licence_metadata_to_candidate():
    candidate = hf_metadata_to_candidate(_meta())
    assert isinstance(candidate, ExternalDatasetCandidate)
    assert candidate.dataset_id == "demo/open-qa"
    assert candidate.licence == "apache-2.0"
    assert candidate.title == "Open QA"
    assert candidate.task_categories == ("question-answering",)
    assert candidate.language == "en"
    assert candidate.publisher == "demo"
    assert candidate.source_url == "https://huggingface.co/datasets/demo/open-qa"
    # A clean, open, well-provenanced, non-gated card can reach knowledge.
    result = assess_hf_metadata(_meta(), now=_NOW)
    assert result.decision == DataIntakeDecision.APPROVED_FOR_KNOWLEDGE
    assert result.lane == IntakeLane.KNOWLEDGE_CANDIDATE


def test_from_dict_reads_card_data_and_tag_encoded_metadata():
    meta = HuggingFaceDatasetMetadata.from_dict({
        "id": "org/name",
        "tags": ["license:mit", "language:en", "task_categories:summarization"],
        "homepage": "https://example.org",
    })
    assert meta.dataset_id == "org/name"
    assert meta.license == "mit"
    assert meta.language == ("en",)
    assert meta.task_categories == ("summarization",)


# 2. missing licence produces a missing_license finding through intake. -------

def test_missing_licence_produces_missing_license_finding():
    result = assess_hf_metadata(_meta(license=None), now=_NOW)
    assert result.decision == DataIntakeDecision.NEEDS_REVIEW
    assert result.approved_for_knowledge is False
    assert any(f.code == FindingCode.MISSING_LICENSE
               for f in result.assessment.findings)


# 3. non-commercial licence is not approved_for_knowledge. -------------------

def test_non_commercial_licence_not_approved_for_knowledge():
    result = assess_hf_metadata(_meta(license="cc-by-nc-4.0"), now=_NOW)
    assert result.approved_for_knowledge is False
    assert result.decision == DataIntakeDecision.APPROVED_FOR_EVAL
    assert any(f.code == FindingCode.NON_COMMERCIAL_LICENSE
               for f in result.assessment.findings)


# 4. gated dataset requires review. ------------------------------------------

def test_gated_dataset_requires_review():
    result = assess_hf_metadata(_meta(gated="manual"), now=_NOW)
    assert result.decision == DataIntakeDecision.NEEDS_REVIEW
    assert result.approved_for_knowledge is False
    assert result.lane != IntakeLane.KNOWLEDGE_CANDIDATE
    assert any("gated" in note for note in result.adapter_notes)


# 5. private dataset requires review or blocked. -----------------------------

def test_private_dataset_requires_review_or_blocked():
    result = assess_hf_metadata(_meta(private=True), now=_NOW)
    assert result.decision in (
        DataIntakeDecision.NEEDS_REVIEW, DataIntakeDecision.BLOCKED)
    assert result.approved_for_knowledge is False
    assert result.lane != IntakeLane.KNOWLEDGE_CANDIDATE


# 6. personal-data tags produce possible_pii or needs_review. ----------------

def test_personal_data_tags_produce_possible_pii_or_review():
    result = assess_hf_metadata(
        _meta(tags=("pii", "personal-data"), contains_personal_data=None),
        now=_NOW)
    codes = {f.code for f in result.assessment.findings}
    assert (FindingCode.POSSIBLE_PII in codes
            or result.decision == DataIntakeDecision.NEEDS_REVIEW)
    assert result.approved_for_knowledge is False


def test_sensitive_domain_tags_require_review():
    result = assess_hf_metadata(
        _meta(tags=("medical", "clinical"), contains_personal_data=None),
        now=_NOW)
    assert result.decision == DataIntakeDecision.NEEDS_REVIEW
    assert result.approved_for_knowledge is False


# 7. synthetic dataset can become eval-only but not knowledge. ---------------

def test_synthetic_dataset_is_eval_only_not_knowledge():
    result = assess_hf_metadata(
        _meta(dataset_id="synth/data", tags=("synthetic",)), now=_NOW)
    assert result.permits_eval_use is True
    assert result.approved_for_knowledge is False
    assert result.lane == IntakeLane.SYNTHETIC_EXAMPLES
    assert result.candidate.is_synthetic is True


# 8. benchmark dataset can become approved_for_eval when licence/prov pass. ---

def test_benchmark_dataset_can_be_approved_for_eval():
    result = assess_hf_metadata(
        _meta(dataset_id="bench/suite", tags=("benchmark",)), now=_NOW)
    assert result.decision == DataIntakeDecision.APPROVED_FOR_EVAL
    assert result.lane == IntakeLane.BENCHMARK
    assert result.approved_for_knowledge is False


# 9. approved_for_eval does not imply approved_for_knowledge. ----------------

def test_approved_for_eval_does_not_imply_knowledge():
    result = assess_hf_metadata(_meta(license="cc-by-nc-4.0"), now=_NOW)
    assert result.permits_eval_use is True
    assert result.approved_for_eval is True
    assert result.approved_for_knowledge is False


# 10. adapter is deterministic. ----------------------------------------------

def test_adapter_is_deterministic():
    a = assess_hf_metadata(_meta(), now=_NOW)
    b = assess_hf_metadata(_meta(), now=_NOW)
    assert a.to_dict() == b.to_dict()
    assert hf_metadata_to_candidate(_meta()) == hf_metadata_to_candidate(_meta())


# 11. the demo metadata file assesses to the expected spread. ----------------

def test_demo_metadata_assesses_as_expected():
    records = load_hf_metadata(_DEMO)
    assert len(records) == 8
    by_id = {r.metadata.dataset_id: r
             for r in hfa.assess_hf_metadata_records(records, now=_NOW)}
    assert by_id["openqa/public-qa"].decision == DataIntakeDecision.APPROVED_FOR_KNOWLEDGE
    assert by_id["synth/eval-instructions"].decision == DataIntakeDecision.APPROVED_FOR_EVAL
    assert by_id["synth/eval-instructions"].lane == IntakeLane.SYNTHETIC_EXAMPLES
    assert by_id["bench/reasoning-suite"].decision == DataIntakeDecision.APPROVED_FOR_EVAL
    assert by_id["bench/reasoning-suite"].lane == IntakeLane.BENCHMARK
    assert by_id["scrape/unlicensed-pairs"].approved_for_knowledge is False
    assert by_id["research/nc-corpus"].approved_for_knowledge is False
    assert by_id["vendor/gated-instructions"].decision == DataIntakeDecision.NEEDS_REVIEW
    assert by_id["forum/user-threads"].decision == DataIntakeDecision.NEEDS_REVIEW
    assert by_id["misc/unknown-origin"].approved_for_knowledge is False
    # No demo dataset is auto-classified as trusted knowledge except the one
    # affirmatively-clean open dataset.
    knowledge = [r for r in by_id.values() if r.approved_for_knowledge]
    assert [r.metadata.dataset_id for r in knowledge] == ["openqa/public-qa"]


# 12. the adapter imports no writer and no dataset-download client. ----------

def test_adapter_imports_no_writer_or_downloader():
    import ast

    source = Path(hfa.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported.add(alias.name)
            if node.module:
                imported.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
    forbidden = {
        "MemoryLedger", "save_registry", "load_registry",
        "save_memory_review_queue", "save_source_review_queue",
        "propose_source_updates", "build_memory_proposals",
        "agent.memory_ledger", "agent.source_registry",
        # dataset download / streaming clients must never be imported
        "requests", "urllib", "urllib.request", "httpx", "datasets",
        "huggingface_hub",
    }
    leaked = imported & forbidden
    assert not leaked, f"hf adapter must not import writers/downloaders: {leaked}"

    body = source.replace(ast.get_docstring(tree, clean=False) or "", "", 1)
    for token in ("MemoryLedger", "save_registry", "propose_source",
                  "memory_review_queue", "source_review_queue",
                  "load_dataset", "hf_hub_download", "snapshot_download",
                  "requests.get", "urlopen"):
        assert token not in body, f"hf adapter must not reference {token!r}"


# 13. no dataset rows are downloaded (no network in the assessment path). -----

def test_assessment_does_no_network(monkeypatch):
    import socket

    def _boom(*args, **kwargs):  # pragma: no cover - only fires on a violation
        raise AssertionError("hf adapter attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", _boom)
    records = load_hf_metadata(_DEMO)
    results = hfa.assess_hf_metadata_records(records, now=_NOW)
    assert len(results) == 8  # assessed purely from declared metadata


# 14. stdout mode writes nothing; --out writes only the assessment report. ----

def test_cli_stdout_mode_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    sys.path.insert(0, str(ROOT / "app"))
    import workbench

    before = sorted(p.name for p in tmp_path.iterdir())
    capsys.readouterr()
    argv = ["hf-data", "assess-metadata", "--metadata", str(_DEMO),
            "--now", "2026-06-04T00:00:00+00:00"]
    assert workbench.main(argv) == 0
    out = capsys.readouterr().out
    assert "# Hugging Face metadata assessment (v6.3; metadata-only)" in out
    assert "Decision:" in out
    after = sorted(p.name for p in tmp_path.iterdir())
    assert before == after  # stdout mode created no files


def test_cli_out_writes_only_the_report(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    sys.path.insert(0, str(ROOT / "app"))
    import workbench

    out = tmp_path / "hf_report.json"
    capsys.readouterr()
    argv = ["hf-data", "assess-metadata", "--metadata", str(_DEMO),
            "--now", "2026-06-04T00:00:00+00:00", "--out", str(out)]
    assert workbench.main(argv) == 0
    created = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert created == [out]  # the report is the only durable write


def test_cli_report_contents_are_assessment_records(tmp_path, monkeypatch, capsys):
    import json

    monkeypatch.chdir(tmp_path)
    sys.path.insert(0, str(ROOT / "app"))
    import workbench

    out = tmp_path / "hf_report.json"
    capsys.readouterr()
    argv = ["hf-data", "assess-metadata", "--metadata", str(_DEMO),
            "--now", "2026-06-04T00:00:00+00:00", "--out", str(out)]
    assert workbench.main(argv) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(payload, list) and len(payload) == 8
    assert all(rec["_record"] == "data_intake_assessment" for rec in payload)


def test_cli_deterministic_output(monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "app"))
    import workbench

    argv = ["hf-data", "assess-metadata", "--metadata", str(_DEMO),
            "--now", "2026-06-04T00:00:00+00:00"]
    capsys.readouterr()
    assert workbench.main(argv) == 0
    first = capsys.readouterr().out
    assert workbench.main(argv) == 0
    second = capsys.readouterr().out
    assert first == second


# 15. README documents the v6.3 Hugging Face adapter. ------------------------

def test_readme_documents_hf_adapter():
    text = _README.read_text(encoding="utf-8")
    assert "## v6.3" in text
    assert "hugging face" in text.lower()
    assert "metadata" in text.lower()


# 16. result.to_dict exposes the metadata, notes, and intake assessment. -----

def test_result_to_dict_shape():
    payload = assess_hf_metadata(_meta(), now=_NOW).to_dict()
    assert payload["_record"] == "hf_metadata_adapter_result"
    assert payload["metadata"]["dataset_id"] == "demo/open-qa"
    assert payload["assessment"]["_record"] == "data_intake_assessment"
    assert isinstance(payload["adapter_notes"], list)
