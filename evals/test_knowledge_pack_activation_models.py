"""v7.0 activation models: pack identity, fingerprints, schema tolerance.

Pins that the generic pack reader tolerates both the Hugging Face and PDF
manifest schemas, that the content fingerprint is deterministic and changes on
any content/order/revision/identity mutation, and that evaluation evidence is
fingerprint-bound.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import knowledge_pack_activation as kpa  # noqa: E402
from _kp_activation_helpers import (  # noqa: E402
    passing_evidence,
    write_hf_pack,
    write_pdf_pack,
)


def test_loads_hf_manifest_schema(tmp_path):
    identity = kpa.load_pack_identity(write_hf_pack(tmp_path / "p"))
    assert identity.source_type == "huggingface"
    assert identity.source_id == "acme/widgets"
    assert identity.source_revision == "rev-aaaa"
    assert identity.authority_level == "reputable"
    assert identity.licence_snapshot == "cc-by-4.0"
    assert identity.pack_kind == "knowledge"
    assert identity.chunk_count == 2
    assert identity.content_fingerprint.startswith(kpa.CONTENT_FP_PREFIX)


def test_loads_pdf_manifest_schema(tmp_path):
    identity = kpa.load_pack_identity(write_pdf_pack(tmp_path / "p"))
    assert identity.source_type == "pdf_document"
    assert identity.source_id == "sha256:pdfaaaa"
    assert identity.source_revision == "previewfp-aaaa"
    assert identity.authority_level == "reputable"
    assert identity.intended_use == "knowledge"
    assert identity.manifest_hash == "packmanifest-pdf-demo"


def test_fingerprint_is_deterministic(tmp_path):
    a = kpa.load_pack_identity(write_hf_pack(tmp_path / "a"))
    b = kpa.load_pack_identity(write_hf_pack(tmp_path / "b"))
    assert a.content_fingerprint == b.content_fingerprint


def test_fingerprint_changes_on_chunk_text(tmp_path):
    base = kpa.load_pack_identity(write_hf_pack(tmp_path / "a"))
    changed = kpa.load_pack_identity(write_hf_pack(
        tmp_path / "b", chunks=[("hfchunk-0001", "alpha body"),
                                ("hfchunk-0002", "CHANGED body")]))
    assert base.content_fingerprint != changed.content_fingerprint


def test_fingerprint_changes_on_chunk_order(tmp_path):
    base = kpa.load_pack_identity(write_hf_pack(tmp_path / "a"))
    reordered = kpa.load_pack_identity(write_hf_pack(
        tmp_path / "b", chunks=[("hfchunk-0002", "beta body"),
                                ("hfchunk-0001", "alpha body")]))
    assert base.content_fingerprint != reordered.content_fingerprint


def test_fingerprint_changes_on_source_revision(tmp_path):
    base = kpa.load_pack_identity(write_hf_pack(tmp_path / "a"))
    bumped = kpa.load_pack_identity(write_hf_pack(
        tmp_path / "b", dataset_revision="rev-bbbb"))
    assert base.content_fingerprint != bumped.content_fingerprint


def test_fingerprint_changes_on_pack_version(tmp_path):
    base = kpa.load_pack_identity(write_hf_pack(tmp_path / "a"))
    bumped = kpa.load_pack_identity(write_hf_pack(
        tmp_path / "b", pack_version="2.0.0"))
    assert base.content_fingerprint != bumped.content_fingerprint


def test_lineage_key_groups_revisions(tmp_path):
    v1 = kpa.load_pack_identity(write_hf_pack(
        tmp_path / "a", pack_id="hf-v1", dataset_revision="rev-1"))
    v2 = kpa.load_pack_identity(write_hf_pack(
        tmp_path / "b", pack_id="hf-v2", dataset_revision="rev-2"))
    assert v1.lineage_key == v2.lineage_key
    assert v1.content_fingerprint != v2.content_fingerprint


def test_eval_pack_kind_is_not_knowledge(tmp_path):
    identity = kpa.load_pack_identity(write_hf_pack(
        tmp_path / "p", pack_kind="eval"))
    assert identity.is_knowledge_pack is False


def test_evidence_fingerprint_deterministic_and_bound(tmp_path):
    identity = kpa.load_pack_identity(write_hf_pack(tmp_path / "p"))
    ev1 = passing_evidence(identity)
    ev2 = passing_evidence(identity)
    assert ev1.compute_fingerprint() == ev2.compute_fingerprint()
    assert ev1.pack_fingerprint == identity.content_fingerprint


def test_evidence_fingerprint_changes_with_counts(tmp_path):
    identity = kpa.load_pack_identity(write_hf_pack(tmp_path / "p"))
    base = passing_evidence(identity)
    from dataclasses import replace
    worse = replace(base, failed_case_count=3)
    assert base.compute_fingerprint() != worse.compute_fingerprint()
