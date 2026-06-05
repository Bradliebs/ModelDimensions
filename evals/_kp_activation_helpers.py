"""Shared fixtures for the knowledge-pack activation tests (not collected)."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from agent import knowledge_pack_activation as kpa  # noqa: E402

FIXED_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def write_hf_pack(pack_dir: Path, *, pack_id: str = "hf-demo",
                  pack_version: str = "1.0.0",
                  dataset_id: str = "acme/widgets",
                  dataset_revision: str = "rev-aaaa",
                  chunks: Optional[Sequence[tuple]] = None,
                  pack_kind: str = "knowledge",
                  authority: str = "reputable",
                  licence: str = "cc-by-4.0",
                  provenance: str = "huggingface") -> Path:
    """Write a Hugging-Face-style knowledge pack to ``pack_dir``."""
    pack_dir.mkdir(parents=True, exist_ok=True)
    chunks = chunks or [("hfchunk-0001", "alpha body"), ("hfchunk-0002", "beta body")]
    knowledge_lines = []
    for chunk_id, text in chunks:
        content_hash = "sha256:" + kpa._sha256_hex(text)
        knowledge_lines.append(json.dumps({
            "_record": "chunk",
            "chunk_id": chunk_id,
            "source_id": dataset_id,
            "chunk_text": text,
            "content_hash": content_hash,
            "active": True,
            "domain": "widgets",
            "authority": authority,
            "source_name": dataset_id,
        }))
    (pack_dir / "knowledge.jsonl").write_text(
        "\n".join(knowledge_lines) + "\n", encoding="utf-8")
    manifest = {
        "_record": "hf_knowledge_pack_manifest",
        "pack_id": pack_id,
        "name": pack_id,
        "description": "demo",
        "created_at": "2024-05-01T00:00:00+00:00",
        "default_knowledge_backend": "deterministic",
        "default_domain": "widgets",
        "settings": {},
        "pack_kind": pack_kind,
        "pack_version": pack_version,
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "split": "train",
        "authority": authority,
        "provenance": provenance,
        "permission_or_licence": licence,
        "chunk_count": len(chunks),
        "importer_version": "hf-knowledge-pack-v6.9",
        "pack_hash": "packhash-hf-demo",
    }
    (pack_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return pack_dir


def write_pdf_pack(pack_dir: Path, *, pack_id: str = "pdf-demo",
                   pack_version: str = "1.0.0",
                   source_file_hash: str = "sha256:pdfaaaa",
                   preview_fingerprint: str = "previewfp-aaaa",
                   chunks: Optional[Sequence[tuple]] = None,
                   authority_level: str = "reputable",
                   licence: str = "internal-use",
                   provenance: str = "internal-pdf") -> Path:
    """Write a PDF-style knowledge pack (different manifest schema)."""
    pack_dir.mkdir(parents=True, exist_ok=True)
    chunks = chunks or [("pdfchunk-0001", "gamma body"), ("pdfchunk-0002", "delta body")]
    knowledge_lines = []
    for chunk_id, text in chunks:
        knowledge_lines.append(json.dumps({
            "_record": "chunk",
            "chunk_id": chunk_id,
            "source_id": source_file_hash,
            "source_type": "pdf_document",
            "chunk_text": text,
            "content_hash": "sha256:" + kpa._sha256_hex(text),
            "active": True,
            "domain": "policies",
            "authority": authority_level,
            "source_name": "policy.pdf",
        }))
    (pack_dir / "knowledge.jsonl").write_text(
        "\n".join(knowledge_lines) + "\n", encoding="utf-8")
    manifest = {
        "_record": "pdf_knowledge_pack_manifest",
        "pack_id": pack_id,
        "name": pack_id,
        "description": "demo",
        "created_at": "2024-05-01T00:00:00+00:00",
        "default_knowledge_backend": "deterministic",
        "default_domain": "policies",
        "settings": {},
        "pack_version": pack_version,
        "source_count": 1,
        "chunk_count": len(chunks),
        "source_file_hash": source_file_hash,
        "preview_fingerprint": preview_fingerprint,
        "approval_id": "pdf-approval-1",
        "approval_actor": "reviewer",
        "approval_timestamp": "2024-05-01T00:00:00+00:00",
        "import_timestamp": "2024-05-01T00:00:00+00:00",
        "intended_use": "knowledge",
        "authority_level": authority_level,
        "provenance": provenance,
        "permission_or_licence": licence,
        "excluded_chunk_ids": [],
        "unresolved_nonblocking_findings": [],
        "importer_version": "pdf-knowledge-pack-v6.6",
        "manifest_hash": "packmanifest-pdf-demo",
    }
    (pack_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return pack_dir


def passing_evidence(identity: kpa.KnowledgePackIdentity, *,
                     report_id: str = "eval-1") -> kpa.KnowledgePackEvaluationEvidence:
    return kpa.KnowledgePackEvaluationEvidence(
        evaluation_report_id=report_id,
        pack_fingerprint=identity.content_fingerprint,
        evaluated_at="2024-05-15T00:00:00+00:00",
        evaluator="evaluator",
        case_count=10,
        passed_case_count=10,
        failed_case_count=0,
        forbidden_source_hit_count=0,
        unrelated_case_regression_count=0,
        unsupported_citation_count=0,
        uncited_factual_claim_count=0,
        expected_source_recall=1.0,
        cited_expected_source_recall=1.0,
        passed=True,
        threshold_policy_id="default-strict-v1",
    )


def approval_for(identity: kpa.KnowledgePackIdentity, *,
                 scope: kpa.ActivationScope = kpa.ActivationScope.ACTIVATE,
                 approval_id: str = "appr-1",
                 evidence: Optional[kpa.KnowledgePackEvaluationEvidence] = None,
                 environment: str = "default",
                 expires_at: str = "",
                 bind_licence: bool = False) -> kpa.KnowledgePackActivationApproval:
    return kpa.KnowledgePackActivationApproval(
        approval_id=approval_id,
        pack_id=identity.pack_id,
        pack_version=identity.pack_version,
        pack_fingerprint=identity.content_fingerprint,
        manifest_hash=identity.manifest_hash,
        approved_by="approver",
        approved_at="2024-05-20T00:00:00+00:00",
        approval_scope=scope,
        approved_environment=environment,
        evaluation_report_id=evidence.evaluation_report_id if evidence else "",
        evaluation_report_fingerprint=evidence.report_fingerprint if evidence else "",
        evaluation_threshold_policy="default-strict-v1",
        expires_at=expires_at,
        licence_snapshot=identity.licence_snapshot if bind_licence else "",
        provenance_snapshot=identity.provenance_snapshot if bind_licence else "",
    )
