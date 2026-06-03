"""Tests for the v1.9 Hugging Face dataset pack importer.

These tests exercise the licence-aware, offline-first importer end to end using
the bundled fixtures in ``demos/hf_fixtures``. Nothing here touches the network:
metadata comes from local JSON cards and rows come from local JSONL fixtures.
The importer reuses the frozen ``KnowledgeLibrary.import_text_file`` path for
knowledge writes and never writes to the ``MemoryLedger``.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent.hf_dataset_importer import (  # noqa: E402
    HFDatasetMode,
    HFDatasetSpec,
    LicensePolicy,
    import_hf_dataset_to_pack,
    sample_dataset,
)
from agent.hf_metadata import fetch_metadata  # noqa: E402
from agent.knowledge_library import KnowledgeLibrary  # noqa: E402
from agent.project_packs import PackRegistry  # noqa: E402
from agent.workbench_service import WorkbenchService  # noqa: E402

_FIXTURES = ROOT / "demos" / "hf_fixtures"
_CARD_WITH_LICENCE = _FIXTURES / "dataset_card_with_license.json"
_CARD_NO_LICENCE = _FIXTURES / "dataset_card_no_license.json"
_CODING_ROWS = _FIXTURES / "small_coding_dataset.jsonl"


def _pack(tmp_path: Path):
    registry = PackRegistry(tmp_path / "packs")
    return registry.create_pack("hf-test", description="hf importer test")


def _knowledge_sources(pack):
    return KnowledgeLibrary(pack.knowledge_library_path).list_sources()


def test_unknown_licence_rejected_for_knowledge_mode_by_default(tmp_path):
    pack = _pack(tmp_path)
    spec = HFDatasetSpec(
        dataset_id="demo/no-license",
        mode=HFDatasetMode.KNOWLEDGE,
        domain="coding",
        authority="reputable",
        card_path=str(_CARD_NO_LICENCE),
        local_fixture=str(_CODING_ROWS),
    )
    report = import_hf_dataset_to_pack(spec, pack)
    assert report.accepted is False
    assert report.imported_count == 0
    assert "licence" in (report.rejected_reason or "").lower()
    assert _knowledge_sources(pack) == []


def test_unknown_licence_allowed_for_eval_mode_with_warning(tmp_path):
    pack = _pack(tmp_path)
    spec = HFDatasetSpec(
        dataset_id="demo/no-license",
        mode=HFDatasetMode.EVAL,
        domain="coding",
        authority="reputable",
        card_path=str(_CARD_NO_LICENCE),
        local_fixture=str(_CODING_ROWS),
    )
    report = import_hf_dataset_to_pack(spec, pack)
    assert report.accepted is True
    assert report.imported_count > 0
    assert any("unknown licence" in w.lower() for w in report.warnings)


def test_sample_size_cap_enforced(tmp_path):
    pack = _pack(tmp_path)
    spec = HFDatasetSpec(
        dataset_id="demo/small-coding",
        mode=HFDatasetMode.EVAL,
        domain="coding",
        authority="reputable",
        sample_size=500,
        card_path=str(_CARD_WITH_LICENCE),
        local_fixture=str(_CODING_ROWS),
    )
    assert spec.effective_sample_size == 100
    report = import_hf_dataset_to_pack(spec, pack)
    assert report.effective_sample_size == 100
    assert report.imported_count <= 100
    assert any("cap" in w.lower() for w in report.warnings)


def test_streaming_default_true():
    spec = HFDatasetSpec(dataset_id="demo/small-coding")
    assert spec.streaming is True


def test_medical_domain_rejects_unknown_authority(tmp_path):
    pack = _pack(tmp_path)
    spec = HFDatasetSpec(
        dataset_id="demo/small-coding",
        mode=HFDatasetMode.EVAL,
        domain="medical",
        authority="unknown",
        card_path=str(_CARD_WITH_LICENCE),
        local_fixture=str(_CODING_ROWS),
    )
    report = import_hf_dataset_to_pack(spec, pack)
    assert report.accepted is False
    assert "authority" in (report.rejected_reason or "").lower()


def test_imported_knowledge_goes_to_library_not_ledger(tmp_path):
    pack = _pack(tmp_path)
    spec = HFDatasetSpec(
        dataset_id="demo/small-coding",
        mode=HFDatasetMode.KNOWLEDGE,
        domain="coding",
        authority="reputable",
        card_path=str(_CARD_WITH_LICENCE),
        local_fixture=str(_CODING_ROWS),
    )
    report = import_hf_dataset_to_pack(spec, pack)
    assert report.accepted is True
    assert report.knowledge_source_id is not None
    assert len(_knowledge_sources(pack)) == 1

    service = WorkbenchService.from_pack(pack)
    ledger = service.export_ledger()
    assert ledger == [] or len(ledger) == 0


def test_eval_mode_writes_eval_samples_not_knowledge(tmp_path):
    pack = _pack(tmp_path)
    spec = HFDatasetSpec(
        dataset_id="demo/small-coding",
        mode=HFDatasetMode.EVAL,
        domain="coding",
        authority="reputable",
        card_path=str(_CARD_WITH_LICENCE),
        local_fixture=str(_CODING_ROWS),
    )
    report = import_hf_dataset_to_pack(spec, pack)
    assert report.accepted is True
    assert report.eval_output_path is not None
    out_path = Path(report.eval_output_path)
    assert out_path.exists()
    lines = [ln for ln in out_path.read_text(encoding="utf-8").splitlines()
             if ln.strip()]
    assert len(lines) == report.imported_count > 0
    assert _knowledge_sources(pack) == []


def test_offline_metadata_failure_is_handled():
    metadata = fetch_metadata("demo/missing", allow_network=False)
    assert metadata.card_present is False
    assert metadata.license_known is False


def test_mocked_dataset_card_licence_is_captured():
    metadata = fetch_metadata("demo/small-coding", card_path=str(_CARD_WITH_LICENCE))
    assert metadata.card_present is True
    assert metadata.license == "apache-2.0"
    assert metadata.license_known is True


def test_path_does_not_silently_download(tmp_path):
    pack = _pack(tmp_path)
    spec = HFDatasetSpec(
        dataset_id="demo/small-coding",
        mode=HFDatasetMode.EVAL,
        domain="coding",
        authority="reputable",
        card_path=str(_CARD_WITH_LICENCE),
        local_fixture=None,
        allow_network=False,
    )
    # The gate would allow this (licensed), but with no fixture and no network
    # opt-in, sampling must refuse rather than download.
    try:
        sample_dataset(spec)
    except RuntimeError as exc:
        assert "download" in str(exc).lower()
    else:  # pragma: no cover - guard against silent download regression
        raise AssertionError("expected a refusal instead of a silent download")
