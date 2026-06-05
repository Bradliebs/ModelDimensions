"""Tests for v6.9 bounded, offline-first Hugging Face row sampling (Phase B).

These prove that :func:`sample_rows` reads at most the approved number of rows,
streams line by line without materialising the whole file, fails closed on an
unapproved split / mismatched revision / missing fixture, is deterministic for a
fixed fixture, and never reaches the network.
"""
from __future__ import annotations

import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import hf_import_lifecycle as life  # noqa: E402
from agent.hf_import_lifecycle import (  # noqa: E402
    HFApprovalScope,
    HFDatasetApproval,
    HFImportIntent,
    HFImportRequest,
    HFSampleStatus,
)

_FIXTURE = ROOT / "demos" / "hf_fixtures" / "governed_qa_sample.jsonl"
_NOW = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)


def _approval(row_limit: int = 100, **overrides) -> HFDatasetApproval:
    base = dict(
        approval_id="hfappr-sample",
        dataset_id="demo/governed-qa",
        dataset_revision="main",
        metadata_fingerprint="hfmeta-04051d615f873f16",
        intake_assessment_fingerprint="",
        approved_by="reviewer@local",
        approved_at="2026-05-02T09:00:00+00:00",
        approval_scope=HFApprovalScope.EVAL_AND_KNOWLEDGE,
        approved_split_names=("train",),
        approved_columns=("id", "question", "answer", "context"),
        row_limit=row_limit,
        streaming_allowed=True,
        approved_intended_use="eval and knowledge",
        licence_snapshot="apache-2.0",
        provenance_snapshot="huggingface:demo/governed-qa (cited, homepage)",
    )
    base.update(overrides)
    return HFDatasetApproval(**base)


def _request(row_limit: int = 100, streaming: bool = True, **overrides) -> HFImportRequest:
    base = dict(
        dataset_id="demo/governed-qa",
        dataset_revision="main",
        split="train",
        intent=HFImportIntent.KNOWLEDGE,
        requested_columns=("id", "question", "answer", "context"),
        requested_row_limit=row_limit,
        streaming=streaming,
    )
    base.update(overrides)
    return HFImportRequest(**base)


def _write_rows(path: Path, count: int) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# generated fixture\n")
        for i in range(count):
            handle.write(json.dumps({"id": f"q{i}", "question": f"q{i}?",
                                     "answer": f"a{i}", "context": "ctx"}) + "\n")


# 1. Sampling stops at the approved cap. -------------------------------------


def test_sampling_stops_at_approved_cap():
    result = life.sample_rows(
        _request(row_limit=2), fixture_path=_FIXTURE,
        approval=_approval(row_limit=2), key_field="id", now=_NOW)
    assert result.status is HFSampleStatus.ROW_LIMIT_REACHED
    assert result.sampled_row_count == 2
    assert result.truncated is True
    assert result.effective_row_limit == 2


# 2. Streaming stops at the approved cap. ------------------------------------


def test_streaming_stops_at_approved_cap():
    result = life.sample_rows(
        _request(row_limit=3, streaming=True), fixture_path=_FIXTURE,
        approval=_approval(row_limit=3), key_field="id", now=_NOW)
    assert result.sampled_row_count == 3
    assert all(r.provenance.streaming is True for r in result.rows)


# 3. The full dataset is never materialised. ---------------------------------


def test_full_dataset_is_not_materialised(tmp_path, monkeypatch):
    big = tmp_path / "big.jsonl"
    _write_rows(big, 1000)

    calls = {"n": 0}
    real_loads = json.loads

    def counting_loads(*args, **kwargs):
        calls["n"] += 1
        return real_loads(*args, **kwargs)

    monkeypatch.setattr(life.json, "loads", counting_loads)
    result = life.sample_rows(
        _request(row_limit=10), fixture_path=big,
        approval=_approval(row_limit=10), key_field="id", now=_NOW)

    assert result.sampled_row_count == 10
    assert result.truncated is True
    # At most cap+1 rows are ever parsed (one extra to detect truncation),
    # never the full 1000-row file.
    assert calls["n"] <= 11


# 4. An unapproved split fails closed (no rows). -----------------------------


def test_unapproved_split_fails_closed():
    result = life.sample_rows(
        _request(split="test"), fixture_path=_FIXTURE,
        approval=_approval(), now=_NOW)
    assert result.status is HFSampleStatus.SAMPLE_BLOCKED
    assert result.sampled_row_count == 0
    assert result.validation is not None and result.validation.valid is False


# 5. A mismatched revision fails closed before any read. ---------------------


def test_unavailable_revision_fails_closed():
    result = life.sample_rows(
        _request(dataset_revision="v2"), fixture_path=_FIXTURE,
        approval=_approval(), now=_NOW)
    assert result.status is HFSampleStatus.SAMPLE_BLOCKED
    assert result.sampled_row_count == 0


# 6. A missing fixture reports source unavailable. ---------------------------


def test_missing_fixture_is_source_unavailable(tmp_path):
    result = life.sample_rows(
        _request(row_limit=5), fixture_path=tmp_path / "nope.jsonl",
        approval=_approval(), now=_NOW)
    assert result.status is HFSampleStatus.SOURCE_UNAVAILABLE
    assert result.sampled_row_count == 0


# 7. Repeated sampling is deterministic for a fixed fixture. -----------------


def test_repeated_sample_is_deterministic():
    a = life.sample_rows(
        _request(row_limit=5), fixture_path=_FIXTURE,
        approval=_approval(row_limit=5), key_field="id", now=_NOW)
    b = life.sample_rows(
        _request(row_limit=5), fixture_path=_FIXTURE,
        approval=_approval(row_limit=5), key_field="id", now=_NOW)
    assert a.to_dict() == b.to_dict()
    assert a.status is HFSampleStatus.SAMPLE_READY


# 8. Each sampled row carries deterministic access provenance. ---------------


def test_sampled_rows_carry_provenance():
    result = life.sample_rows(
        _request(row_limit=5), fixture_path=_FIXTURE,
        approval=_approval(row_limit=5), key_field="id", now=_NOW)
    first = result.rows[0]
    assert first.provenance.dataset_id == "demo/governed-qa"
    assert first.provenance.dataset_revision == "main"
    assert first.provenance.split == "train"
    assert first.provenance.source_row_index == 0
    assert first.provenance.source_row_key == "q1"
    assert first.provenance.row_content_hash.startswith("sha256:")
    assert first.provenance.schema_fingerprint.startswith("hfschema-")
    assert first.provenance.adapter_version == life.ADAPTER_VERSION


# 9. A schema change across rows is flagged. ---------------------------------


def test_schema_mismatch_is_flagged(tmp_path):
    mixed = tmp_path / "mixed.jsonl"
    with mixed.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"id": "a", "question": "q", "answer": "x",
                                 "context": "c"}) + "\n")
        handle.write(json.dumps({"id": "b", "different": "shape"}) + "\n")
    result = life.sample_rows(
        _request(row_limit=10), fixture_path=mixed,
        approval=_approval(row_limit=10,
                           approved_columns=("id", "question", "answer",
                                             "context", "different")),
        now=_NOW)
    assert result.status is HFSampleStatus.SCHEMA_MISMATCH


# 10. The markdown summary never prints raw row content. ---------------------


def test_markdown_summary_excludes_raw_content():
    result = life.sample_rows(
        _request(row_limit=5), fixture_path=_FIXTURE,
        approval=_approval(row_limit=5), key_field="id", now=_NOW)
    md = life.render_sample_result_markdown(result)
    # A distinctive answer fragment must not leak into the summary.
    assert "error-minutes" not in md
    assert "demo/governed-qa" in md


# 11. write_sample_result is the only durable write and round-trips. ---------


def test_write_sample_result_round_trips(tmp_path):
    result = life.sample_rows(
        _request(row_limit=5), fixture_path=_FIXTURE,
        approval=_approval(row_limit=5), key_field="id", now=_NOW)
    out = life.write_sample_result(result, tmp_path / "sample.json")
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["_record"] == "hf_sample_result"
    assert payload["sampled_row_count"] == 5


# 12. Sampling never touches the network. ------------------------------------


def test_sampling_makes_no_network_calls(monkeypatch):
    def boom(*args, **kwargs):  # pragma: no cover - must never be called
        raise AssertionError("network access attempted during sampling")

    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    result = life.sample_rows(
        _request(row_limit=5), fixture_path=_FIXTURE,
        approval=_approval(row_limit=5), key_field="id", now=_NOW)
    assert result.ready is True
