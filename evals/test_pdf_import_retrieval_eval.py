"""Tests for the v6.7 imported-PDF retrieval evaluation harness (read-only).

These tests pin the contract that measures whether an approved imported PDF
knowledge pack actually improves retrieval, source selection, and final
report/chat citations — without introducing wrong-source bleed, off-topic
inclusion, citation drift, or regression against an empty baseline.

The harness is strictly a measurement layer: it traces every case through the
frozen read path (``query_knowledge`` -> ``build_grounding_package`` ->
``answer_query`` with the consultant report composer) and scores source / chunk
/ page lineage at all three stages. It changes no retrieval, ranking, chunking,
grounding, composer, or memory behaviour, and writes nothing it is not asked to.

PDF packs are imported programmatically with the v6.6 importer (fixed ``now`` for
determinism); services are built read-only from the imported pack directories.
"""
from __future__ import annotations

import ast
import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import retrieval_eval_harness as reh  # noqa: E402
from agent.pdf_chunk_preview import preview_pdf_chunks  # noqa: E402
from agent.pdf_pack_importer import (  # noqa: E402
    PdfImportApproval,
    PdfImportRequest,
    compute_preview_fingerprint,
    import_pdf_pack,
)
from agent.project_packs import PackRegistry  # noqa: E402
from agent.workbench_service import WorkbenchService  # noqa: E402
from retrieval.embedding_backend import OfflineHashingEmbedder  # noqa: E402

_README = ROOT / "README.md"
_NOW = datetime(2026, 6, 4, tzinfo=timezone.utc)
_IMPORT_NOW = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)

_QUARTERLY = "Quarterly Operating Review"
_ANNUAL = "Annual Reliability Summary"

# Distinctive single-fact pages (sentence-terminated, long enough for a "good"
# extraction band; each fact lives on exactly one page for clean page lineage).
_Q_PAGES = [
    ("The platform reliability budget ceiling is fixed at 4200 error-minutes "
     "for every quarter and the capacity planning team watches throughput and "
     "latency against that ceiling. Each service owner confirms the 4200 "
     "error-minute reliability budget before approving any production rollout."),
    ("The named accountable owner for capacity planning is Dana Okafor, who "
     "chairs the weekly operating review and signs off the agreed follow up "
     "actions. Dana Okafor is recorded as the single accountable owner for the "
     "platform reliability programme for the current quarter."),
    ("Any reliability regression beyond the agreed threshold triggers a "
     "documented investigation within five business days and a written "
     "remediation plan. The investigation records the threshold breach, the "
     "affected services, and the corrective actions taken by the team."),
]
_A_PAGES = [
    ("The annual reliability summary reports a yearly downtime budget of 9000 "
     "error-minutes across all services and reviews throughput and latency "
     "trends over twelve months. The annual figure of 9000 error-minutes is a "
     "retrospective total rather than a quarterly reliability ceiling."),
]


# --------------------------------------------------------------------------- #
# Deterministic single-literal PDF fixture writer (matches the v6.4 reader).
# --------------------------------------------------------------------------- #


def _escape(text: str) -> str:
    return (text.replace("\\", "\\\\").replace("(", "\\(")
            .replace(")", "\\)").replace("\n", "\\n"))


def _text_stream(text: str) -> str:
    return "\n".join(["BT", "/F1 12 Tf", "72 720 Td",
                      f"({_escape(text)}) Tj", "ET"])


def _make_pdf(pages, *, title) -> bytes:
    n_pages = len(pages)
    content_start = 3
    page_start = content_start + n_pages
    info_num = page_start + n_pages
    page_nums = [page_start + i for i in range(n_pages)]
    chunks = [b"%PDF-1.4\n"]

    def obj(num: int, body: str) -> None:
        chunks.append(f"{num} 0 obj\n{body}\nendobj\n".encode("latin-1"))

    obj(1, "<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{p} 0 R" for p in page_nums)
    obj(2, f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>")
    for i, page_text in enumerate(pages):
        cnum = content_start + i
        pnum = page_start + i
        content = _text_stream(page_text)
        obj(cnum, f"<< /Length {len(content)} >>\nstream\n{content}\nendstream")
        obj(pnum, f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                  f"/Contents {cnum} 0 R >>")
    obj(info_num, f"<< /Title ({title}) /Author (Platform Team) "
                  "/Creator (ModelDimensions) /Producer (ModelDimensions) "
                  "/CreationDate (D:20250101000000Z) >>")
    chunks.append(
        f"trailer\n<< /Root 1 0 R /Info {info_num} 0 R >>\n".encode("latin-1"))
    chunks.append(b"%%EOF\n")
    return b"".join(chunks)


# --------------------------------------------------------------------------- #
# Import a deterministic PDF pack and build a read-only service from it.
# --------------------------------------------------------------------------- #


def _import_pack(tmp_path: Path, pages, *, pack_id: str, source_title: str,
                 parent: Path | None = None) -> Path:
    pdf_path = tmp_path / f"{pack_id}.pdf"
    pdf_path.write_bytes(_make_pdf(pages, title=source_title))
    preview = preview_pdf_chunks(
        pdf_path, source_url="https://example.org/report.pdf",
        owner="Platform Team", permission="cc-by-4.0",
        intended_use="knowledge_candidate", authority_level="official",
        now=_NOW).to_dict(include_full_text=True)
    chunk_ids = [c["chunk_id"] for c in preview["chunks"]]
    approval = PdfImportApproval(
        approval_id=f"appr-{pack_id}", approved_by="reviewer@local",
        approved_at="2026-06-04T00:00:00+00:00",
        source_file_hash=preview["file_hash"],
        preview_fingerprint=compute_preview_fingerprint(preview),
        approved_chunk_ids=tuple(chunk_ids),
        intended_pack_id=pack_id, intended_use="knowledge_candidate",
        source_title=source_title, authority_level="official",
        provenance="declared by document owner",
        permission_or_licence="cc-by-4.0", domain="general")
    pack_dir = (parent or (tmp_path / "packs")) / pack_id
    result = import_pdf_pack(
        PdfImportRequest(preview=preview, approval=approval, pack_id=pack_id,
                         pack_dir=str(pack_dir), write=True),
        now=_IMPORT_NOW)
    assert result.written, result.status
    return pack_dir


def _service(pack_dir: Path) -> WorkbenchService:
    registry = PackRegistry(pack_dir.parent)
    pack = registry.get_pack(pack_dir.name)
    assert pack is not None
    return WorkbenchService.from_pack(
        pack, registry=registry, knowledge_backend="hybrid",
        semantic_embedder=OfflineHashingEmbedder())


def _empty_service(tmp_path: Path) -> WorkbenchService:
    return WorkbenchService(
        knowledge_path=str(tmp_path / "empty_knowledge.jsonl"),
        knowledge_backend="hybrid", semantic_embedder=OfflineHashingEmbedder())


def _combined_service(tmp_path: Path, pack_dirs) -> WorkbenchService:
    """Build a read-only service over several packs' knowledge concatenated.

    A pure read: the source packs are never modified; their knowledge records
    are concatenated byte-for-byte into a throwaway file used only for this
    service (needed to exercise near-neighbour source competition).
    """
    combined = tmp_path / "combined_knowledge.jsonl"
    blocks = []
    for pd in pack_dirs:
        text = (pd / "knowledge.jsonl").read_text(encoding="utf-8").strip()
        if text:
            blocks.append(text)
    combined.write_text("\n".join(blocks) + "\n", encoding="utf-8")
    return WorkbenchService(
        knowledge_path=str(combined), knowledge_backend="hybrid",
        semantic_embedder=OfflineHashingEmbedder())


def _quarterly_pack(tmp_path: Path, **kw) -> Path:
    return _import_pack(tmp_path, _Q_PAGES, pack_id="quarterly_review",
                        source_title=_QUARTERLY, **kw)


# -- inline cases -------------------------------------------------------------

_BUDGET = reh.PdfImportRetrievalCase(
    case_id="budget", pack_id="quarterly_review",
    query="What is the platform reliability budget ceiling of 4200 "
          "error-minutes for the quarter?",
    expected_source_ids=[_QUARTERLY], expected_page_ranges=[[1, 1]],
    expected_terms=["4200"], pdf_only=True, tags=["pdf", "direct"])

_OWNER = reh.PdfImportRetrievalCase(
    case_id="owner", pack_id="quarterly_review",
    query="Who is the named accountable owner for capacity planning, "
          "Dana Okafor, at the weekly operating review?",
    expected_source_ids=[_QUARTERLY], expected_page_ranges=[[2, 2]],
    expected_terms=["Okafor"], pdf_only=True, tags=["pdf", "decision"])

_GAP = reh.PdfImportRetrievalCase(
    case_id="gap",
    query="What is the marketing campaign budget for next year?",
    expected_gap=True, expected_answer_mode="refusal", tags=["pdf", "gap"])

_UNRELATED = reh.PdfImportRetrievalCase(
    case_id="unrelated",
    query="How do I rotate an Azure storage account access key?",
    unrelated=True, forbidden_source_ids=[_QUARTERLY],
    expected_answer_mode="refusal", tags=["pdf", "unrelated"])

_NEIGHBOUR = reh.PdfImportRetrievalCase(
    case_id="neighbour", pack_id="quarterly_review",
    query="What is the quarterly reliability budget ceiling of 4200 "
          "error-minutes?",
    expected_source_ids=[_QUARTERLY], expected_terms=["4200"],
    minimum_hit_k=2, tags=["pdf", "neighbour"])


# --------------------------------------------------------------------------- #
# 1. A PDF-only answer becomes retrievable once the pack is enabled.
# --------------------------------------------------------------------------- #


def test_pdf_only_answer_retrievable_with_pack(tmp_path):
    pack_dir = _quarterly_pack(tmp_path)
    base = reh.evaluate_pdf_import_case(_empty_service(tmp_path), _BUDGET)
    withpack = reh.evaluate_pdf_import_case(_service(pack_dir), _BUDGET)

    assert base.retrieval.retrieved_chunk_count == 0
    assert base.citation.final_answer_grounded is False
    assert withpack.retrieval.rank_of_first_expected_source is not None
    assert withpack.citation.final_answer_grounded is True
    assert withpack.passed is True


# 2. The expected source is retrieved within top-k. --------------------------


def test_expected_source_within_top_k(tmp_path):
    res = reh.evaluate_pdf_import_case(_service(_quarterly_pack(tmp_path)),
                                       _BUDGET)
    assert res.retrieval.hit_at_5 is True
    assert res.retrieval.expected_source_recall == 1.0
    assert res.retrieval.rank_of_first_expected_source <= _BUDGET.minimum_hit_k


# 3. Expected page lineage is preserved through retrieval. -------------------


def test_expected_page_lineage_preserved(tmp_path):
    res = reh.evaluate_pdf_import_case(_service(_quarterly_pack(tmp_path)),
                                       _BUDGET)
    # The page-1 fact must carry page lineage (page_start == page_end == 1).
    budget_items = [it for it in res.raw.items if "4200" in it.text]
    assert budget_items, "expected the 4200 budget chunk to be retrieved"
    assert all(it.page_start == 1 and it.page_end == 1 for it in budget_items)
    assert res.retrieval.expected_page_recall == 1.0


# 4. A forbidden / unrelated query selects no imported PDF evidence. ----------


def test_unrelated_query_selects_no_pdf_evidence(tmp_path):
    res = reh.evaluate_pdf_import_case(_service(_quarterly_pack(tmp_path)),
                                       _UNRELATED)
    # Isolation is enforced at the relevance gate: nothing is selected or cited.
    assert res.selected.items == []
    assert res.citation.cited_chunk_count == 0
    assert res.citation.cited_forbidden_source_count == 0
    assert res.refused is True
    assert res.passed is True


# 5. Near-neighbour source competition is scored correctly. ------------------


def test_near_neighbour_competition(tmp_path):
    quarterly = _quarterly_pack(tmp_path)
    annual = _import_pack(tmp_path, _A_PAGES, pack_id="annual_summary",
                          source_title=_ANNUAL)
    service = _combined_service(tmp_path, [quarterly, annual])
    res = reh.evaluate_pdf_import_case(service, _NEIGHBOUR)

    # Both sources are present in the corpus, but the expected (quarterly, 4200)
    # source must be retrieved; the competing annual source is not forbidden.
    names = res.raw.source_names
    assert _QUARTERLY in names
    assert res.retrieval.expected_source_recall == 1.0
    top = res.raw.items[0]
    assert "4200" in top.text and _QUARTERLY.lower() in (top.source_name or "").lower()


# 6. Selected evidence preserves the expected source. ------------------------


def test_selected_evidence_preserves_expected_source(tmp_path):
    res = reh.evaluate_pdf_import_case(_service(_quarterly_pack(tmp_path)),
                                       _BUDGET)
    assert res.selected.items, "expected on-topic evidence to pass the gate"
    assert res.selected_metrics.selected_expected_source_recall == 1.0
    assert res.selected_metrics.selected_forbidden_source_count == 0


# 7. The final citation preserves the expected source and chunk lineage. -----


def test_final_citation_preserves_source_and_page(tmp_path):
    res = reh.evaluate_pdf_import_case(_service(_quarterly_pack(tmp_path)),
                                       _BUDGET)
    cited = [it for it in res.final.items if it.selection_reason == "cited"]
    assert cited, "expected the grounded report to cite evidence"
    assert res.citation.cited_expected_source_recall == 1.0
    # Every cited item resolves to a page in the source (lineage preserved).
    assert all(it.page_start is not None for it in cited)


# 8. The citation set is a subset of the selected evidence. ------------------


def test_citation_subset_of_selected(tmp_path):
    res = reh.evaluate_pdf_import_case(_service(_quarterly_pack(tmp_path)),
                                       _OWNER)
    assert res.citation.citation_set_matches_selected_evidence is True
    assert res.citation.unsupported_citation_count == 0
    assert res.citation.uncited_factual_claim_count == 0


# 9. An insufficient-evidence query stays insufficient (no grounded answer). --


def test_insufficient_evidence_stays_insufficient(tmp_path):
    res = reh.evaluate_pdf_import_case(_service(_quarterly_pack(tmp_path)),
                                       _GAP)
    assert res.citation.final_answer_grounded is False
    assert res.passed is True
    assert res.classification == reh.PDF_INSUFFICIENT_EVIDENCE_CORRECT


# 10-12. Baseline-vs-pack comparison is a pure diff. -------------------------


def _retr(*, hit5, forbidden=0, wrong=0.0):
    return reh.PdfRetrievalMetrics(
        retrieved_chunk_count=(1 if hit5 else 0),
        hit_at_1=hit5, hit_at_3=hit5, hit_at_5=hit5,
        expected_source_recall=(1.0 if hit5 else 0.0),
        expected_chunk_recall=None, expected_page_recall=None,
        wrong_source_rate=wrong, forbidden_source_hit_count=forbidden,
        forbidden_chunk_hit_count=0, off_topic_inclusion_rate=0.0,
        duplicate_chunk_rate=0.0, near_duplicate_chunk_rate=0.0,
        missing_expected_source_count=0, missing_expected_chunk_count=0,
        rank_of_first_expected_source=(1 if hit5 else None),
        rank_of_first_expected_chunk=None)


def _sel(*, forbidden=0, n=1):
    return reh.PdfSelectedMetrics(
        selected_chunk_count=n, selected_expected_source_recall=1.0,
        selected_expected_chunk_recall=None, selected_wrong_source_rate=0.0,
        selected_forbidden_source_count=forbidden,
        selected_off_topic_inclusion_rate=0.0,
        selected_page_lineage_accuracy=None)


def _cit(*, grounded, forbidden=0, cite_recall=1.0):
    return reh.PdfCitationMetrics(
        cited_chunk_count=(1 if grounded else 0),
        cited_expected_source_recall=(cite_recall if grounded else 0.0),
        cited_expected_chunk_recall=None, cited_wrong_source_rate=0.0,
        cited_forbidden_source_count=forbidden,
        citation_page_lineage_accuracy=None, uncited_factual_claim_count=0,
        unsupported_citation_count=0,
        citation_set_matches_selected_evidence=True,
        final_answer_grounded=grounded)


def _result(case_id, *, grounded, hit5=True, forbidden=0, tags=(),
            selected_items=()):
    return reh.PdfImportCaseResult(
        case_id=case_id, query="q", pack_id="p",
        passed=(grounded and not forbidden), reasons=[],
        classification=reh.PDF_NO_FAILURE,
        answer_mode=("grounded" if grounded else "refusal"),
        refused=(not grounded),
        raw=reh.PdfStageTrace("raw", []),
        selected=reh.PdfStageTrace("selected", list(selected_items)),
        final=reh.PdfStageTrace("final", []),
        retrieval=_retr(hit5=hit5, forbidden=forbidden),
        selected_metrics=_sel(forbidden=forbidden),
        citation=_cit(grounded=grounded, forbidden=forbidden),
        tags=list(tags))


def test_compare_detects_improvement():
    base = [_result("c1", grounded=False, hit5=False)]
    pack = [_result("c1", grounded=True, hit5=True)]
    cmp = reh.compare_pdf_import_eval(base, pack)
    assert cmp.cases_improved == 1
    assert cmp.cases_regressed == 0
    assert cmp.newly_answerable_cases == 1
    assert cmp.pack_helps is True


def test_compare_detects_regression():
    # Baseline answered the case; enabling the pack breaks grounding.
    base = [_result("c1", grounded=True, hit5=True)]
    pack = [_result("c1", grounded=False, hit5=False)]
    cmp = reh.compare_pdf_import_eval(base, pack)
    assert cmp.cases_regressed == 1
    assert cmp.cases_improved == 0
    assert "c1" in cmp.regressed_case_ids
    assert cmp.pack_helps is False


def test_compare_flags_unrelated_case_affected():
    leaked = reh.PdfStageItem("selected", "chk-leak", source_name="Imported PDF")
    base = [_result("u1", grounded=False, hit5=False, tags=("unrelated",))]
    pack = [_result("u1", grounded=False, hit5=False, tags=("unrelated",),
                    selected_items=[leaked])]
    cmp = reh.compare_pdf_import_eval(base, pack)
    assert cmp.unrelated_cases_affected == 1
    assert "u1" in cmp.unrelated_affected_case_ids


# 13. Repeated runs are deterministic. ---------------------------------------


def test_repeated_runs_deterministic(tmp_path):
    pack_dir = _quarterly_pack(tmp_path)
    cases = [_BUDGET, _OWNER, _GAP, _UNRELATED]
    run1 = [r.to_dict() for r in
            reh.run_pdf_import_eval(_service(pack_dir), cases)]
    run2 = [r.to_dict() for r in
            reh.run_pdf_import_eval(_service(pack_dir), cases)]
    assert run1 == run2


# 14. The evaluation leaves the imported pack byte-identical. ----------------


def test_imported_pack_byte_identical_after_eval(tmp_path):
    pack_dir = _quarterly_pack(tmp_path)
    before = {p.name: p.read_bytes() for p in pack_dir.iterdir() if p.is_file()}
    reh.run_pdf_import_eval(_service(pack_dir), [_BUDGET, _OWNER, _UNRELATED])
    assert before["manifest.json"] == (pack_dir / "manifest.json").read_bytes()
    assert before["knowledge.jsonl"] == (pack_dir / "knowledge.jsonl").read_bytes()


# 15. The evaluation writes no registry / proposal / review-queue artefacts. --


def test_eval_writes_no_durable_state(tmp_path):
    pack_dir = _quarterly_pack(tmp_path)
    reh.run_pdf_import_eval(_service(pack_dir), [_BUDGET, _GAP, _UNRELATED])
    names = {p.name for p in pack_dir.rglob("*") if p.is_file()}
    for forbidden in ("source_registry.jsonl", "memory_proposals.jsonl",
                      "source_proposals.jsonl", "review_queue.jsonl"):
        assert forbidden not in names


# 16. Summary + comparison shapes are stable and serialisable. ---------------


def test_summary_and_comparison_shapes(tmp_path):
    pack_dir = _quarterly_pack(tmp_path)
    cases = [_BUDGET, _OWNER, _GAP, _UNRELATED]
    results = reh.run_pdf_import_eval(_service(pack_dir), cases)
    summary = reh.summarize_pdf_import_eval(results)
    assert summary.case_count == len(cases)
    assert summary.pass_count + summary.fail_count == len(cases)
    # Raw recall is unfiltered by design; true isolation is the absence of any
    # forbidden source in the selected evidence and final citations.
    assert summary.cited_forbidden_source_total == 0
    assert summary.forbidden_bleed_reproduced is False
    assert sum(summary.classification_counts.values()) == len(cases)

    base = reh.run_pdf_import_eval(_empty_service(tmp_path), cases)
    cmp = reh.compare_pdf_import_eval(base, results)
    assert cmp.case_count == len(cases)
    assert cmp.unrelated_cases_affected == 0
    # Markdown rendering is pure and mentions the read-only contract.
    md = reh.render_pdf_import_markdown(
        results, summary, pack_label="quarterly_review",
        backend_label="hybrid", comparison=cmp)
    assert "Imported PDF retrieval evaluation" in md
    assert "Read-only" in md


# 17. CLI stdout mode prints the report and writes nothing. ------------------


def test_cli_stdout_prints_report(tmp_path, monkeypatch, capsys):
    pack_dir = _quarterly_pack(tmp_path)
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(json.dumps(_BUDGET.to_dict()) + "\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    sys.path.insert(0, str(ROOT / "app"))
    import workbench

    before = sorted(p.name for p in tmp_path.iterdir())
    capsys.readouterr()
    argv = ["retrieval-eval", "pdf-import", "--cases", str(cases_path),
            "--pack", str(pack_dir)]
    assert workbench.main(argv) == 0
    out = capsys.readouterr().out
    assert "Imported PDF retrieval evaluation" in out
    after = sorted(p.name for p in tmp_path.iterdir())
    assert before == after  # stdout mode created no files in the cwd


# 18. CLI --out writes only the report and is silent on stdout. --------------


def test_cli_out_writes_only_report(tmp_path, capsys):
    pack_dir = _quarterly_pack(tmp_path)
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(json.dumps(_BUDGET.to_dict()) + "\n", encoding="utf-8")
    sys.path.insert(0, str(ROOT / "app"))
    import workbench

    out = tmp_path / "out" / "pdf_import_report.md"
    capsys.readouterr()
    argv = ["retrieval-eval", "pdf-import", "--cases", str(cases_path),
            "--pack", str(pack_dir), "--compare-without-pack", "--out", str(out)]
    assert workbench.main(argv) == 0
    assert capsys.readouterr().out == ""  # silent when writing to --out
    assert out.exists()
    created = [p for p in (tmp_path / "out").rglob("*") if p.is_file()]
    assert created == [out]
    assert "Imported PDF retrieval evaluation" in out.read_text(encoding="utf-8")


# 19. The harness imports no durable-state writer (AST + token purity). ------


def test_harness_imports_no_writer():
    source = Path(reh.__file__).read_text(encoding="utf-8")
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
        "save_registry", "save_source_registry", "propose_source_updates",
        "build_memory_proposals", "save_memory_review_queue",
        "save_source_review_queue", "import_pdf_pack",
        "agent.source_registry", "agent.source_proposals",
        "agent.memory_proposals", "agent.pdf_pack_importer",
        "pytesseract", "easyocr", "openai", "anthropic",
    }
    leaked = imported & forbidden
    assert not leaked, f"harness must not import writers/OCR/LLM: {leaked}"

    code = source.replace(ast.get_docstring(tree, clean=False) or "", "", 1)
    for token in ("save_registry", "propose_source", "memory_review_queue",
                  "source_review_queue", "pytesseract", "easyocr",
                  "import_pdf_pack(", "openai", "anthropic"):
        assert token not in code, f"harness must not reference {token!r}"


# 20. The harness opens no network socket while evaluating. ------------------


def test_harness_opens_no_socket(tmp_path, monkeypatch):
    def _boom(*_a, **_k):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", _boom)
    pack_dir = _quarterly_pack(tmp_path)
    reh.run_pdf_import_eval(_service(pack_dir), [_BUDGET, _UNRELATED])


# 21. The shipped demo corpus loads and is well-formed. ----------------------


def test_demo_corpus_loads():
    path = ROOT / "demos" / "retrieval_pdf_import_cases.jsonl"
    cases = reh.load_pdf_import_cases(path)
    assert len(cases) >= 14
    ids = {c.case_id for c in cases}
    assert {"direct_factual_lookup", "insufficient_evidence",
            "forbidden_unrelated_source"} <= ids
    # Every case is internally consistent.
    for c in cases:
        if c.unrelated or c.expected_gap:
            assert not c.expects_pdf_content
        if c.expects_pdf_content:
            assert c.expected_source_ids


# 22. README documents the v6.7 slice. ---------------------------------------


def test_readme_documents_v67():
    text = _README.read_text(encoding="utf-8")
    assert "## v6.7" in text
    assert "Imported PDF Retrieval Evaluation" in text
