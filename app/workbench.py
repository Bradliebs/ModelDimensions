"""Concept Memory Workbench (v1.1).

A small, fully offline local app for working with the frozen v1.0 memory:
write, query, verify, refuse, delete, inspect, and export — every action showing
a clear audit trail (candidate retrieved, verifier verdict, memory used,
refused).

It adds no architecture. All behaviour comes from ``WorkbenchService``, which
composes the untouched v1.0 path (retrieve -> verify -> ground).

Run as a CLI (default, no extra dependencies):

    python app/workbench.py                 # interactive REPL (seeds the project)
    python app/workbench.py --demo          # run the Friday -> Monday example
    python app/workbench.py --no-seed       # start with an empty workbench

Or, if Streamlit is installed, as a local web app:

    streamlit run app/workbench.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent.workbench_service import QueryAudit, WorkbenchService  # noqa: E402
from agent.project_packs import PackRegistry  # noqa: E402
from agent import pack_builder  # noqa: E402
from agent import pack_evaluator  # noqa: E402
from agent import hf_dataset_importer  # noqa: E402
_DEFAULT_LEDGER = ROOT / "demos" / "workbench_ledger.jsonl"
_DEFAULT_QUEUE = ROOT / "demos" / "workbench_proposals.jsonl"
_SEED_FILE = ROOT / "demos" / "seed_concept_cells_project.jsonl"
_INGEST_NOTE = ROOT / "demos" / "seed_ingestion_note.md"
_DEFAULT_PACK_ROOT = ROOT / "demos" / "packs"

# The canonical Friday -> Monday example: a stored fact and a one-word near-miss.
_FRIDAY_FACT = "the supplier delivery is on Friday afternoon"
_MONDAY_QUERY = "the supplier delivery is on Monday afternoon"


# ---------- shared rendering ----------

def _format_audit(audit: QueryAudit) -> str:
    lines = [
        f'Query: "{audit.query}"',
        f"  candidate_retrieved = {str(audit.candidate_retrieved).lower()}",
    ]
    for cand in audit.candidates:
        lines.append(
            f"    - {cand.memory_id} "
            f"(rank {cand.rank}, activation {cand.activation:.4f}) "
            f"-> {cand.verdict.upper()}"
        )
    lines.append(f"  verifier_verdict    = {audit.verifier_verdict.upper()}")
    lines.append(f"  memory_used         = {str(audit.memory_used).lower()}")
    lines.append(f"  refused             = {str(audit.refused).lower()}")
    if audit.cited_memory_ids:
        lines.append(f"  cited_memory_ids    = {', '.join(audit.cited_memory_ids)}")
    if audit.historical:
        lines.append("  historical (superseded — not a current answer):")
        for h in audit.historical:
            sup = f" -> {h['superseded_by']}" if h.get("superseded_by") else ""
            lines.append(
                f"    - {h['memory_id']} (overlap {h['overlap']:.4f}){sup}: "
                f"{h['canonical_text']}"
            )
    lines.append("  response:")
    for row in audit.response_text.splitlines():
        lines.append(f"    {row}")
    return "\n".join(lines)


def _parse_flags(arg: str) -> tuple[str, dict]:
    """Split ``<positional> --flag value ...`` into (positional, {flag: value}).

    Used by ``import-knowledge`` to read ``--domain``/``--authority``/``--name``/
    ``--version``. The positional is everything before the first ``--`` flag.
    """
    tokens = arg.split()
    positional: list[str] = []
    flags: dict[str, str] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--"):
            key = tok[2:]
            value_parts: list[str] = []
            i += 1
            while i < len(tokens) and not tokens[i].startswith("--"):
                value_parts.append(tokens[i])
                i += 1
            flags[key] = " ".join(value_parts)
        else:
            positional.append(tok)
            i += 1
    return " ".join(positional), flags


def _format_knowledge(audit) -> str:
    lines = [
        f'Knowledge query: "{audit.query}"',
        f"  knowledge_used      = {str(audit.knowledge_used).lower()}",
        f"  backend             = {getattr(audit, 'backend_name', 'deterministic')}",
        f"  domain              = {audit.domain or '-'}",
        f"  informational_only  = {str(audit.informational_only).lower()}",
    ]
    for cand in audit.candidates:
        section = f" [{cand['section']}]" if cand.get("section") else ""
        lines.append(
            f"    - {cand['source_name']}{section} "
            f"(rank {cand['rank']}, activation {cand['activation']:.4f}, "
            f"{cand['authority']})"
        )
    if audit.cited_source_ids:
        lines.append(f"  cited_source_ids    = {', '.join(audit.cited_source_ids)}")
    for caution in audit.cautions:
        lines.append(f"  ! {caution}")
    lines.append("  response:")
    for row in audit.response_text.splitlines():
        lines.append(f"    {row}")
    return "\n".join(lines)


def _format_combined(audit) -> str:
    lines = [
        f'Query: "{audit.query}"',
        f"  route               = {audit.route}",
        f"  memory_used         = {str(audit.memory_used).lower()}",
        f"  knowledge_used      = {str(audit.knowledge_used).lower()}",
        f"  model_prior_used    = {str(audit.model_prior_used).lower()}",
    ]
    if audit.memory:
        lines.append("  -- project memory --")
        for cid in audit.memory.get("cited_memory_ids") or []:
            lines.append(f"    cited memory: {cid}")
        if not (audit.memory.get("cited_memory_ids")):
            lines.append("    (memory silent)")
    if audit.knowledge:
        lines.append("  -- imported knowledge --")
        for cand in audit.knowledge.get("candidates") or []:
            lines.append(f"    knowledge: {cand['source_name']} "
                         f"({cand['authority']})")
        if not (audit.knowledge.get("candidates")):
            lines.append("    (knowledge silent)")
    for caution in audit.cautions:
        lines.append(f"  ! {caution}")
    return "\n".join(lines)


def _format_assistant(result) -> str:
    """Render an AssistantResult: the composed answer plus its audit trail."""
    answer = result.answer
    lines = [
        f'Assisted answer: "{result.query}"',
        f"  composer_backend    = {result.composer_backend}",
        f"  mode                = {answer.mode.value}",
        f"  refused             = {str(result.refused).lower()}",
        f"  fell_back_to_template = {str(answer.fell_back).lower()}",
        f"  memory_used         = {str(result.audit.get('memory_used')).lower()}",
        f"  knowledge_used      = "
        f"{str(result.audit.get('knowledge_used')).lower()}",
        f"  model_prior_used    = "
        f"{str(result.audit.get('model_prior_used')).lower()}",
    ]
    relevance = result.audit.get("relevance") or {}
    if relevance:
        lines.append(f"  relevance           = {relevance.get('verdict', '-')} "
                     f"({relevance.get('label', '-')})")
        reason = relevance.get("sufficiency_reason")
        if reason:
            lines.append(f"  sufficiency         = {reason}")
        rejected = relevance.get("top_rejected") or {}
        if rejected.get("citation_id"):
            lines.append(f"  top_rejected        = {rejected['citation_id']} — "
                         f"{rejected.get('reason', '')}")
    if result.evidence_ids:
        lines.append(f"  evidence_ids        = {', '.join(result.evidence_ids)}")
    if answer.citations:
        lines.append(f"  cited               = {', '.join(answer.citations)}")
    if answer.informational_only:
        lines.append("  informational_only  = true")
    lines.append("  answer:")
    for row in answer.text.splitlines():
        lines.append(f"    {row}")
    return "\n".join(lines)


# ---------- CLI ----------

_HELP = """\
Commands:
  add <text>            write a new memory
  query <text>          retrieve -> verify -> ground (shows the audit trail)
  delete <memory_id>    delete a memory (it can no longer be cited)
  ledger                show the memory ledger (active / deleted)
  import <path>         extract candidate memories from a note (queued, not written)
  proposals             show pending candidate memories awaiting review
  conflicts             show pending candidates that conflict with a memory
  duplicates            show pending candidates that duplicate a memory
  analyse <proposal_id> re-check a candidate against the current memories
  approve <proposal_id> approve a candidate so write-approved will store it
  approve-new <proposal_id>  force-approve a flagged candidate as a new memory
  approve-supersede <proposal_id> <old_memory_id>  approve and replace an old memory
  reject <proposal_id>  reject a candidate (it is never written)
  edit <proposal_id> <text>  edit a candidate's text before approving
  write-approved        write only approved candidates into the ledger
  queue-export [path]   write the proposal queue JSONL
  show-memory <memory_id>  show a memory and its supersession history
  query-history <text>     query, also surfacing superseded memories as history
  import-knowledge <path> --domain <d> --authority <a> --name <n> [--version <v>]
                        import an external doc into the knowledge library
  sources               list imported knowledge sources
  query-knowledge <text>  query imported knowledge only (with domain cautions)
  query-all <text>      query memory and knowledge, kept clearly separated
  query-assist <text>   compose a readable answer over the audit (template)
  query-assist --slm <text>  same, using the optional local SLM (falls back safely)
  query-assist --report <text>  same, structured as a deterministic consultant report
  backend               show the active knowledge retrieval backend
  packs                 list project packs (isolated workspaces)
  pack-create <name>    create a new project pack
  pack-use <name>       switch the active pack (re-points all stores); alias: pack-switch
  pack-info [name]      show the active pack (or a named pack) and its paths
  pack-sources [--stale] [--json]  list source freshness (current/review_due/stale/unknown)
  pack-refresh-plan [--json]  list sources needing review, with risk and action
  pack-maintenance-report [--json]  summarise pack source health
  pack-export <name> <path>  export a pack to a .zip bundle
  pack-import <path>    import a pack bundle and register it
  pack-build <spec>     build a curated knowledge pack from a JSON/YAML spec
  pack-eval <pack> <eval-file>  validate a pack's retrieval with eval questions
  pack-report <pack>    show a pack's curated sources and chunk counts
  retrieval-eval [--pack p] [--backend hybrid]  measure retrieval quality (read-only baseline)
  retrieval-eval --probe-report-path  localise where relevance bleed enters the report path (read-only)
  retrieval-eval --probe-hygiene  classify wrong-source candidates on a harder near-neighbour corpus (read-only)
  source-registry list [--registry path]  list source metadata entries (read-only)
  source-registry inspect <id> [--registry path]  show one source's metadata (read-only)
  source-registry audit [--registry path]  report source lifecycle/metadata risk (read-only)
  source-registry propose-updates [--registry path] [--out path]  generate source-maintenance proposals (read-only; not applied)
  source-registry proposal-review import-proposals <proposals.jsonl> [--queue path]  queue proposals for review (review-state only; not applied)
  source-registry proposal-review list [--queue path]  list proposal review state (read-only)
  source-registry proposal-review review <id> --status approved|rejected|deferred [--note t] [--reviewer r] [--queue path]  record a review decision (approved != applied)
  memory-proposals list [candidates.jsonl]  show typed, evidence-bound memory proposals for review (read-only; no memory written)
  memory-proposals build [candidates.jsonl] [--out path]  build typed memory proposals (markdown, or JSONL via --out; no memory written)
  hf-inspect <dataset_id>  preview a Hugging Face dataset's licence/decision
  hf-import <dataset_id> --pack <pack>  import a small governed HF sample
  demo                  run the Friday -> Monday near-miss example
  export [path]         write the ledger JSONL (defaults to the active ledger)
  help                  show this help
  quit                  exit
"""


def _print_ledger(service: WorkbenchService) -> None:
    rows = service.export_ledger()
    if not rows:
        print("  (ledger is empty)")
        return
    for row in rows:
        tags = ",".join(row.get("tags") or [])
        line = (f"  {row['memory_id']}  [{row['status']}]  "
                f"{row['canonical_text']}")
        if tags:
            line += f"   (tags: {tags})"
        print(line)


def _print_proposals(service: WorkbenchService, rows=None,
                     empty_msg=None) -> None:
    if rows is None:
        rows = service.list_proposals(status="pending")
    if not rows:
        print(empty_msg or "  (no pending proposals — import a note first)")
        return
    for p in rows:
        tags = ",".join(p.tags) if p.tags else ""
        loc = f"L{p.source_line_start}" if p.source_line_start else "?"
        line = (f"  {p.proposal_id}  [{p.kind.value}] "
                f"(conf {p.confidence:.2f}, {loc})  {p.canonical_text}")
        if tags:
            line += f"   (tags: {tags})"
        print(line)
        if p.lifecycle_verdict and p.lifecycle_verdict != "new":
            against = (f" vs {p.lifecycle_candidate_id}"
                       if p.lifecycle_candidate_id else "")
            print(f"      lifecycle: {p.lifecycle_verdict.upper()}{against} "
                  f"— {p.lifecycle_reason}")
        elif p.reason:
            print(f"      why: {p.reason}")


def _show_memory(service: WorkbenchService, memory_id: str) -> None:
    entry = service.ledger.get(memory_id)
    if entry is None:
        print(f"  unknown memory: {memory_id}")
        return
    print(f"  {entry.memory_id}  [{entry.status}]  {entry.canonical_text}")
    if entry.tags:
        print(f"      tags: {', '.join(entry.tags)}")
    if entry.supersedes:
        print(f"      supersedes: {', '.join(entry.supersedes)}")
    if entry.superseded_by:
        current = service.ledger.get_current_memory(memory_id)
        print(f"      superseded_by: {entry.superseded_by} "
              f"(current: {current.memory_id if current else '?'})")


def _run_demo() -> None:
    # The example runs on its own fresh, in-memory workbench so the Friday cell
    # is the only memory present: the Monday query then retrieves exactly that
    # cell and the verifier rejects it on the weekday flip. This keeps the
    # demonstration crisp and leaves the project ledger untouched.
    service = WorkbenchService(ledger_path=None)
    print("\n=== Friday -> Monday near-miss example ===")
    entry = service.add_memory(_FRIDAY_FACT, source="demo", tags=["example"])
    print(f'Stored {entry.memory_id}: "{_FRIDAY_FACT}"')

    print("\nExact query (expect ACCEPT -> grounded):")
    print(_format_audit(service.query_memory(_FRIDAY_FACT)))

    print("\nNear-miss query — one word changed, Friday -> Monday "
          "(expect REJECT -> NOT grounded):")
    print(_format_audit(service.query_memory(_MONDAY_QUERY)))

    print(f"\nDelete {entry.memory_id}, then query again "
          "(expect silent refusal):")
    service.delete_memory(entry.memory_id)
    print(_format_audit(service.query_memory(_FRIDAY_FACT)))
    print("\nA retrieved candidate is not a true fact: the verifier, not "
          "retrieval, decides grounding.\n")


def _pack_label(service: WorkbenchService) -> str:
    """A short ``[pack: name]`` prefix for audit output, or '' if global."""
    info = service.active_pack_info()
    return f"[pack: {info['name']}] " if info else ""


def _seed_pack_if_empty(service: WorkbenchService, pack) -> None:
    """Seed a freshly-used pack from its manifest's seed settings, if present."""
    if not service.export_ledger():
        mem_seed = pack.resolve_setting_path("seed_memory")
        if mem_seed is not None:
            service.seed_from(mem_seed)
    if not service.list_knowledge_sources():
        kn_seed = pack.resolve_setting_path("seed_knowledge")
        if kn_seed is not None:
            service.import_knowledge(
                kn_seed,
                domain=str(pack.settings.get("knowledge_domain", "general")),
                authority=str(pack.settings.get("knowledge_authority", "unknown")),
                source_name=str(pack.settings.get("knowledge_name", pack.name)),
            )


def _print_packs(registry: PackRegistry) -> None:
    packs = registry.list_packs()
    if not packs:
        print("  (no packs yet — create one with: pack-create <name>)")
        return
    active = registry.get_active_pack()
    active_id = active.pack_id if active else None
    for pack in packs:
        marker = "*" if pack.pack_id == active_id else " "
        desc = f"  — {pack.description}" if pack.description else ""
        print(f"  {marker} {pack.pack_id}  ({pack.name}){desc}")


def _print_pack_info(info: dict) -> None:
    print(f"  pack_id   : {info['pack_id']}")
    print(f"  name      : {info['name']}")
    if info.get("description"):
        print(f"  desc      : {info['description']}")
    print(f"  root      : {info['root_path']}")
    print(f"  memories  : {info['memory_entries']} ledger entries")
    print(f"  proposals : {info['proposal_entries']} queued")
    print(f"  knowledge : {info['knowledge_records']} records")
    print(f"  backend   : {info['default_knowledge_backend']}")


def _print_inventory(rows: list) -> None:
    if not rows:
        print("  (no knowledge sources in the active pack)")
        return
    for r in rows:
        age = f"{r.age_days}d" if r.age_days is not None else "?"
        print(f"  [{r.status.value:<10}] {r.source_name}  "
              f"({r.domain}/{r.authority}, {r.entries} entr"
              f"{'y' if r.entries == 1 else 'ies'}, age {age})")
        print(f"               policy={r.staleness_policy}  — {r.reason}")


def _print_refresh_plan(items: list) -> None:
    if not items:
        print("  all sources are current — nothing to refresh.")
        return
    for it in items:
        print(f"  [{it.risk:<6}] {it.source_name}  ({it.status.value})")
        print(f"             why : {it.reason}")
        print(f"             do  : {it.suggested_action}")


def _print_maintenance_report(report) -> None:
    print(f"  sources    : {report.source_count} "
          f"({report.total_entries} entries)")
    print(f"  current    : {report.current_count}")
    print(f"  review_due : {report.review_due_count}")
    print(f"  stale      : {report.stale_count}")
    print(f"  unknown    : {report.unknown_count}")
    if report.eval_total:
        rate = (report.eval_pass_rate or 0.0) * 100.0
        print(f"  eval       : {report.eval_passed}/{report.eval_total} "
              f"passed ({rate:.1f}%)")
    else:
        print("  eval       : not run in this session "
              "(use scripts/build_m365_coding_pack.py for the eval gate)")



def _print_build_report(report: "pack_builder.PackBuildReport") -> None:
    print(f"  pack      : {report.pack_name} ({report.pack_id})")
    print(f"  sources   : {report.source_count} "
          f"({report.total_chunks} chunks)")
    for src in report.sources:
        print(f"    - {src['source_name']} [{src['domain']}/{src['authority']}"
              f"{(' v' + src['version']) if src.get('version') else ''}] "
              f"-> {src['chunks']} chunks")
    if report.skipped:
        print(f"  skipped   : {len(report.skipped)}")
        for skip in report.skipped:
            print(f"    - {skip['source_name']} ({skip['reason']})")


def _print_eval_results(results: list) -> None:
    passed = sum(1 for r in results if r.passed)
    for res in results:
        mark = "PASS" if res.passed else "FAIL"
        label = res.question_id or res.query
        print(f"  [{mark}] {label}")
        if not res.passed:
            print(f"         {res.reason}")
    print(f"  summary   : {passed}/{len(results)} passed")


def _repl(service: WorkbenchService,
          registry: PackRegistry | None = None) -> None:
    label = _pack_label(service).strip()
    suffix = f" {label}" if label else ""
    print("Concept Memory Workbench (v1.1) — offline. "
          f"Type 'help' for commands.{suffix}")

    while True:
        try:
            raw = input("workbench> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            continue
        parts = raw.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in {"quit", "exit"}:
            break
        elif cmd == "help":
            print(_HELP)
        elif cmd == "add":
            if not arg:
                print("  usage: add <text>")
                continue
            entry = service.add_memory(arg, source="workbench")
            print(f"  stored {entry.memory_id}")
        elif cmd == "query":
            if not arg:
                print("  usage: query <text>")
                continue
            label = _pack_label(service).strip()
            if label:
                print(label)
            print(_format_audit(service.query_memory(arg)))
        elif cmd in {"query-history", "query_history"}:
            if not arg:
                print("  usage: query-history <text>")
                continue
            label = _pack_label(service).strip()
            if label:
                print(label)
            print(_format_audit(
                service.query_memory(arg, include_historical=True)))
        elif cmd == "delete":
            if not arg:
                print("  usage: delete <memory_id>")
                continue
            ok = service.delete_memory(arg)
            print(f"  deleted {arg}: {str(ok).lower()}")
        elif cmd == "ledger":
            _print_ledger(service)
        elif cmd == "import":
            if not arg:
                print("  usage: import <path>")
                continue
            try:
                batch = service.import_notes(arg)
            except FileNotFoundError:
                print(f"  no such file: {arg}")
                continue
            print(f"  imported {len(batch)} candidate(s) from {arg} "
                  "(queued as pending; nothing written yet)")
        elif cmd == "proposals":
            _print_proposals(service)
        elif cmd == "conflicts":
            _print_proposals(service, service.list_conflicts(),
                             "  (no pending conflicts)")
        elif cmd == "duplicates":
            _print_proposals(service, service.list_duplicates(),
                             "  (no pending duplicates)")
        elif cmd == "analyse":
            if not arg:
                print("  usage: analyse <proposal_id>")
                continue
            check = service.analyse_proposal(arg)
            if check is None:
                print(f"  unknown proposal: {arg}")
            else:
                against = (f" vs {check.candidate_memory_id}"
                           if check.candidate_memory_id else "")
                print(f"  {arg}: {check.verdict.value.upper()}{against} "
                      f"— {check.reason}")
        elif cmd == "approve":
            if not arg:
                print("  usage: approve <proposal_id>")
                continue
            ok = service.approve_proposal(arg)
            if ok:
                print(f"  approved {arg}: true")
            else:
                print(f"  approved {arg}: false (flagged duplicate/conflict — "
                      "use approve-new or approve-supersede)")
        elif cmd in {"approve-new", "approve_new"}:
            if not arg:
                print("  usage: approve-new <proposal_id>")
                continue
            ok = service.approve_proposal_as_new(arg)
            print(f"  approved (as new) {arg}: {str(ok).lower()}")
        elif cmd in {"approve-supersede", "approve_supersede"}:
            sup_parts = arg.split(maxsplit=1)
            if len(sup_parts) < 2:
                print("  usage: approve-supersede <proposal_id> <old_memory_id>")
                continue
            ok = service.approve_proposal_superseding(
                sup_parts[0], sup_parts[1].strip())
            if ok:
                print(f"  approved {sup_parts[0]} superseding "
                      f"{sup_parts[1].strip()}: true")
            else:
                print(f"  approved-supersede {sup_parts[0]}: false "
                      "(unknown proposal or old memory id)")
        elif cmd == "reject":
            if not arg:
                print("  usage: reject <proposal_id>")
                continue
            ok = service.reject_proposal(arg)
            print(f"  rejected {arg}: {str(ok).lower()}")
        elif cmd == "edit":
            edit_parts = arg.split(maxsplit=1)
            if len(edit_parts) < 2:
                print("  usage: edit <proposal_id> <new text>")
                continue
            ok = service.edit_proposal(edit_parts[0], edit_parts[1].strip())
            print(f"  edited {edit_parts[0]}: {str(ok).lower()}")
        elif cmd in {"write-approved", "write_approved"}:
            written = service.write_approved_proposals()
            if not written:
                print("  no approved candidates to write")
            else:
                for entry in written:
                    print(f"  wrote {entry.memory_id}: "
                          f"\"{entry.canonical_text}\"")
        elif cmd in {"queue-export", "queue_export"}:
            target = Path(arg) if arg else service.proposals.path
            if target is None:
                print("  no queue path set; usage: queue-export <path>")
                continue
            service.proposals.export_to(target)
            print(f"  exported proposal queue to {target}")
        elif cmd in {"show-memory", "show_memory"}:
            if not arg:
                print("  usage: show-memory <memory_id>")
                continue
            _show_memory(service, arg)
        elif cmd in {"import-knowledge", "import_knowledge"}:
            positional, flags = _parse_flags(arg)
            if not positional or "domain" not in flags or \
                    "authority" not in flags or "name" not in flags:
                print("  usage: import-knowledge <path> --domain <d> "
                      "--authority <a> --name <n> [--version <v>]")
                continue
            try:
                source = service.import_knowledge(
                    positional,
                    domain=flags["domain"],
                    authority=flags["authority"],
                    source_name=flags["name"],
                    version=flags.get("version"),
                )
            except FileNotFoundError:
                print(f"  no such file: {positional}")
                continue
            except ValueError as exc:
                print(f"  invalid domain/authority: {exc}")
                continue
            n_chunks = len(service.knowledge.list_chunks(
                source_id=source.source_id))
            print(f"  imported '{source.source_name}' as {source.source_id} "
                  f"({n_chunks} chunk(s)); not written as project memory")
        elif cmd == "sources":
            rows = service.list_knowledge_sources()
            if not rows:
                print("  (no imported knowledge sources)")
            else:
                for row in rows:
                    ver = row.get("version") or "-"
                    print(f"  {row['source_id']}  {row['source_name']}  "
                          f"[{row['domain']}/{row['authority']}] "
                          f"v={ver}  chunks={row['chunks']}")
        elif cmd in {"query-knowledge", "query_knowledge"}:
            if not arg:
                print("  usage: query-knowledge <text>")
                continue
            print(_format_knowledge(service.query_knowledge(arg)))
        elif cmd in {"query-all", "query_all"}:
            if not arg:
                print("  usage: query-all <text>")
                continue
            print(_format_combined(service.query_all(arg)))
        elif cmd in {"query-assist", "query_assist"}:
            use_slm = False
            composer = None
            text = arg
            if text.startswith("--slm"):
                use_slm = True
                text = text[len("--slm"):].strip()
            elif text.startswith("--report"):
                from slm.assistant_composer import ConsultantReportComposer
                composer = ConsultantReportComposer()
                text = text[len("--report"):].strip()
            if not text:
                print("  usage: query-assist [--slm|--report] <text>")
                continue
            print(_format_assistant(
                service.answer_query(text, use_slm=use_slm,
                                     composer=composer)))
        elif cmd == "backend":
            name = service.knowledge_backend_name()
            print(f"  knowledge retrieval backend = {name}")
            if name == "deterministic":
                print("  (offline/test mode — reproducible, not semantic)")
                print("  enable semantic retrieval with: "
                      "set KNOWLEDGE_RETRIEVAL_BACKEND=semantic")
            else:
                print("  (semantic mode — meaning-based; retrieval is not truth)")
        elif cmd == "demo":
            _run_demo()
        elif cmd in {"packs", "pack-list", "pack_list"}:
            if registry is None:
                print("  packs are disabled; relaunch with --pack-root <dir>")
                continue
            _print_packs(registry)
        elif cmd in {"pack-create", "pack_create"}:
            if registry is None:
                print("  packs are disabled; relaunch with --pack-root <dir>")
                continue
            if not arg:
                print("  usage: pack-create <name>")
                continue
            pack = registry.create_pack(arg)
            print(f"  created pack {pack.pack_id} ({pack.name})")
        elif cmd in {"pack-use", "pack_use", "pack-switch", "pack_switch"}:
            if registry is None:
                print("  packs are disabled; relaunch with --pack-root <dir>")
                continue
            if not arg:
                print("  usage: pack-use <name>")
                continue
            try:
                pack = service.switch_pack(arg)
            except (KeyError, ValueError) as exc:
                print(f"  {exc}")
                continue
            _seed_pack_if_empty(service, pack)
            print(f"  active pack -> {pack.pack_id} ({pack.name})")
        elif cmd in {"pack-info", "pack_info"}:
            if registry is None:
                print("  packs are disabled; relaunch with --pack-root <dir>")
                continue
            if arg:
                info = registry.pack_info(arg)
            else:
                info = service.active_pack_info()
            if info is None:
                print("  no active pack (using global stores)" if not arg
                       else f"  unknown pack: {arg}")
                continue
            _print_pack_info(info)
        elif cmd in {"pack-sources", "pack_sources"}:
            if registry is None:
                print("  packs are disabled; relaunch with --pack-root <dir>")
                continue
            flags = arg.split()
            rows = service.source_inventory()
            if "--stale" in flags:
                rows = [r for r in rows if r.status.value != "current"]
            if "--json" in flags:
                print(json.dumps([r.to_dict() for r in rows], indent=2))
            else:
                _print_inventory(rows)
        elif cmd in {"pack-refresh-plan", "pack_refresh_plan"}:
            if registry is None:
                print("  packs are disabled; relaunch with --pack-root <dir>")
                continue
            items = service.refresh_plan()
            if "--json" in arg.split():
                print(json.dumps([i.to_dict() for i in items], indent=2))
            else:
                _print_refresh_plan(items)
        elif cmd in {"pack-maintenance-report", "pack_maintenance_report"}:
            if registry is None:
                print("  packs are disabled; relaunch with --pack-root <dir>")
                continue
            report = service.maintenance_report()
            if "--json" in arg.split():
                print(json.dumps(report.to_dict(), indent=2))
            else:
                _print_maintenance_report(report)
        elif cmd in {"pack-export", "pack_export"}:
            if registry is None:
                print("  packs are disabled; relaunch with --pack-root <dir>")
                continue
            bits = arg.split(maxsplit=1)
            if len(bits) != 2:
                print("  usage: pack-export <name> <path>")
                continue
            try:
                out = registry.export_pack(bits[0], bits[1].strip())
            except KeyError as exc:
                print(f"  {exc}")
                continue
            print(f"  exported pack to {out}")
        elif cmd in {"pack-import", "pack_import"}:
            if registry is None:
                print("  packs are disabled; relaunch with --pack-root <dir>")
                continue
            if not arg:
                print("  usage: pack-import <path>")
                continue
            try:
                pack = registry.import_pack(arg)
            except (ValueError, OSError) as exc:
                print(f"  {exc}")
                continue
            print(f"  imported pack {pack.pack_id} ({pack.name})")
        elif cmd in {"pack-build", "pack_build"}:
            if registry is None:
                print("  packs are disabled; relaunch with --pack-root <dir>")
                continue
            if not arg:
                print("  usage: pack-build <spec.json|spec.yaml>")
                continue
            try:
                plan = pack_builder.PackBuildPlan.from_file(arg.strip())
            except (OSError, ValueError, RuntimeError) as exc:
                print(f"  {exc}")
                continue
            if not plan.enabled:
                print(f"  spec {plan.pack_name} is disabled (enabled: false); "
                      "not building")
                continue
            report = pack_builder.build_pack(plan, registry)
            _print_build_report(report)
        elif cmd in {"pack-eval", "pack_eval"}:
            if registry is None:
                print("  packs are disabled; relaunch with --pack-root <dir>")
                continue
            bits = arg.split(maxsplit=1)
            if len(bits) != 2:
                print("  usage: pack-eval <pack> <eval-file.jsonl>")
                continue
            try:
                eval_service = pack_builder.open_pack_service(registry, bits[0])
                results = pack_evaluator.evaluate_pack_file(
                    eval_service, bits[1].strip())
            except (KeyError, OSError, ValueError) as exc:
                print(f"  {exc}")
                continue
            _print_eval_results(results)
        elif cmd in {"pack-report", "pack_report"}:
            if registry is None:
                print("  packs are disabled; relaunch with --pack-root <dir>")
                continue
            if not arg:
                print("  usage: pack-report <pack>")
                continue
            pack = registry.get_pack(arg.strip())
            if pack is None:
                print(f"  unknown pack: {arg.strip()}")
                continue
            report_service = pack_builder.open_pack_service(
                registry, arg.strip())
            report = pack_builder.report_for_pack(
                report_service, pack_id=pack.pack_id, pack_name=pack.name)
            _print_build_report(report)
        elif cmd in {"hf-inspect", "hf_inspect"}:
            if registry is None:
                print("  packs are disabled; relaunch with --pack-root <dir>")
                continue
            positional, flags = _parse_flags(arg)
            dataset_id = positional.strip()
            if not dataset_id:
                print("  usage: hf-inspect <dataset_id> [--card <card.json>] "
                      "[--fixture <rows.jsonl>] [--mode knowledge|eval] "
                      "[--domain <domain>] [--authority <authority>]")
                continue
            spec = hf_dataset_importer.HFDatasetSpec(
                dataset_id=dataset_id,
                mode=flags.get("mode") or "eval",
                domain=flags.get("domain") or "general",
                authority=flags.get("authority") or "unknown",
                card_path=flags.get("card") or None,
                local_fixture=flags.get("fixture") or None,
            )
            result = hf_dataset_importer.inspect_dataset(spec)
            meta = result["metadata"]
            decision = result["decision"]
            print(f"  dataset      = {dataset_id}")
            print(f"  card_present = {str(meta['card_present']).lower()}")
            print(f"  license      = {meta['license'] or '-'} "
                  f"(known: {str(meta['license_known']).lower()})")
            print(f"  would_import = {str(decision['would_import']).lower()}")
            if decision["rejected_reason"]:
                print(f"  blocked      = {decision['rejected_reason']}")
            for warn in decision["warnings"]:
                print(f"  warning      = {warn}")
        elif cmd in {"hf-import", "hf_import"}:
            if registry is None:
                print("  packs are disabled; relaunch with --pack-root <dir>")
                continue
            positional, flags = _parse_flags(arg)
            dataset_id = positional.strip()
            pack_name = flags.get("pack", "").strip()
            if not dataset_id or not pack_name:
                print("  usage: hf-import <dataset_id> --pack <pack> "
                      "[--split <split>] [--field <field>] "
                      "[--sample-size <n>] [--mode knowledge|eval] "
                      "[--domain <domain>] [--authority <authority>] "
                      "[--card <card.json>] [--fixture <rows.jsonl>]")
                continue
            pack = registry.get_pack(pack_name)
            if pack is None:
                print(f"  unknown pack: {pack_name}")
                continue
            fields = [flags["field"]] if flags.get("field") else ["text"]
            try:
                sample_size = (int(flags["sample-size"])
                               if flags.get("sample-size") else 100)
            except ValueError:
                print("  --sample-size must be an integer")
                continue
            spec = hf_dataset_importer.HFDatasetSpec(
                dataset_id=dataset_id,
                split=flags.get("split") or "train",
                text_fields=fields,
                sample_size=sample_size,
                mode=flags.get("mode") or "eval",
                domain=flags.get("domain") or "general",
                authority=flags.get("authority") or "unknown",
                card_path=flags.get("card") or None,
                local_fixture=flags.get("fixture") or None,
            )
            try:
                report = hf_dataset_importer.import_hf_dataset_to_pack(
                    spec, pack)
            except RuntimeError as exc:
                print(f"  refused: {exc}")
                continue
            print(f"  dataset       = {report.dataset_id}")
            print(f"  mode          = {report.mode}")
            print(f"  accepted      = {str(report.accepted).lower()}")
            if not report.accepted:
                print(f"  blocked       = {report.rejected_reason}")
            else:
                print(f"  imported      = {report.imported_count} rows")
                if report.knowledge_source_id:
                    print(f"  knowledge_src = {report.knowledge_source_id}")
                if report.eval_output_path:
                    print(f"  eval_samples  = {report.eval_output_path}")
            for warn in report.warnings:
                print(f"  warning       = {warn}")
        elif cmd == "export":
            target = Path(arg) if arg else service.ledger.path
            if target is None:
                print("  no ledger path set; usage: export <path>")
                continue
            service.ledger.path = Path(target)
            service.ledger._flush()  # write-through to the chosen path
            print(f"  exported ledger to {target}")
        else:
            print(f"  unknown command: {cmd} (type 'help')")


# ---------- v2.3 value sprint CLI ----------

_VALUE_SPRINT_MANIFEST_DIR = ROOT / "packs"
_VALUE_SPRINT_QUERIES = ROOT / "demos" / "value_sprint_queries.jsonl"
_VALUE_SPRINT_DIR = ROOT / "reports"


def _build_sprint_service(pack_name: str, backend: str, *, seed: bool):
    """Build the named pack into a temporary registry and bind a service.

    The pack is built into a throwaway temp directory so the tracked pack data
    stays pristine, and the demo project memories are seeded into that temporary
    ledger only (never a tracked store) so the memory-routed paths can fire.
    """
    import tempfile

    from agent.value_sprint_harness import DEMO_SEED_MEMORIES

    manifest = _VALUE_SPRINT_MANIFEST_DIR / pack_name / "pack.yaml"
    if not manifest.exists():
        raise SystemExit(f"no pack manifest at {manifest}")

    tmp_root = Path(tempfile.mkdtemp(prefix="value_sprint_"))
    registry = PackRegistry(tmp_root / "packs")
    plan = pack_builder.PackBuildPlan.from_file(manifest)
    report = pack_builder.build_pack(plan, registry)
    pack = registry.get_pack(report.pack_id)

    embedder = None
    if backend == "hybrid":
        from retrieval.embedding_backend import OfflineHashingEmbedder
        embedder = OfflineHashingEmbedder()
    service = WorkbenchService.from_pack(
        pack, registry=registry,
        knowledge_backend=backend, semantic_embedder=embedder)

    if seed:
        for text in DEMO_SEED_MEMORIES:
            service.add_memory(text, source="value-sprint-demo")
    return service


def _value_sprint_cli(argv: list[str]) -> int:
    from agent import value_sprint_harness as vsh
    from slm.assistant_composer import (
        ConsultantReportComposer,
        ExtractiveMultiChunkComposer,
    )

    parser = argparse.ArgumentParser(
        prog="workbench.py value-sprint",
        description="Run the v2.3 value sprint or re-render its report.")
    sub = parser.add_subparsers(dest="action", required=True)

    run = sub.add_parser("run", help="run the sprint and write reports")
    run.add_argument("--pack", default="m365_coding_assistant",
                     help="pack manifest directory under packs/")
    run.add_argument("--backend", default="deterministic",
                     choices=["deterministic", "hybrid"],
                     help="knowledge retrieval backend (default deterministic)")
    run.add_argument("--composer", default="template",
                     choices=["template", "extractive", "report"],
                     help="answer composer: 'template' echoes whole chunks "
                          "(default); 'extractive' quotes the most relevant "
                          "verbatim span from each source; 'report' structures "
                          "the evidence as a deterministic consultant report "
                          "(factual sections cited, judgement sections "
                          "labelled and uncited)")
    run.add_argument("--queries", default=str(_VALUE_SPRINT_QUERIES),
                     help="path to the value sprint query JSONL")
    run.add_argument("--no-seed", dest="seed", action="store_false",
                     default=True,
                     help="do not seed the demo project memories")
    run.add_argument("--out-md", default=str(_VALUE_SPRINT_DIR
                                             / "value_sprint_latest.md"))
    run.add_argument("--out-jsonl", default=str(_VALUE_SPRINT_DIR
                                                / "value_sprint_latest.jsonl"))
    run.add_argument("--emit-memory-proposals", dest="emit_proposals",
                     action="store_true", default=False,
                     help="opt-in: route missing-decision pack gaps into a "
                          "sprint-scoped proposal queue (PENDING only, never "
                          "written; default off)")
    run.add_argument("--proposals-queue",
                     default=str(_VALUE_SPRINT_DIR
                                 / "value_sprint_proposals.jsonl"),
                     help="sprint-scoped proposal queue path (only written when "
                          "--emit-memory-proposals is set)")

    rep = sub.add_parser("report",
                         help="re-render Markdown from a sprint JSONL report")
    rep.add_argument("--jsonl", default=str(_VALUE_SPRINT_DIR
                                            / "value_sprint_latest.jsonl"))
    rep.add_argument("--out-md", default=str(_VALUE_SPRINT_DIR
                                             / "value_sprint_latest.md"))

    args = parser.parse_args(argv)

    if args.action == "run":
        service = _build_sprint_service(args.pack, args.backend, seed=args.seed)
        queries = vsh.load_queries(args.queries)
        if args.composer == "extractive":
            composer = ExtractiveMultiChunkComposer()
        elif args.composer == "report":
            composer = ConsultantReportComposer()
        else:
            composer = None
        rows = vsh.run_sprint(service, queries, retrieval_backend=args.backend,
                              composer=composer)
        summary = vsh.summarize(rows)
        vsh.write_reports(
            rows, summary, md_path=args.out_md, jsonl_path=args.out_jsonl,
            pack_label=args.pack, backend_label=args.backend)
        print(f"[value-sprint] {summary.query_count} queries | "
              f"grounded={summary.grounded_count} "
              f"refused={summary.refused_count} "
              f"conflicts={summary.conflict_count} "
              f"model-prior={summary.model_prior_count} "
              f"stale={summary.stale_flagged_count} "
              f"gaps={summary.pack_gap_count} "
              f"guard-rejects={summary.guard_reject_count}")
        print(f"[value-sprint] composer={args.composer} "
              f"multi-source-grounded={summary.multi_source_grounded_count} "
              f"spans={sum(r.span_count for r in rows)}")
        if args.composer == "report":
            print("[value-sprint] "
                  f"report-cited-claims={sum(r.report_cited_claim_count for r in rows)} "
                  f"report-judgement-blocks={sum(r.report_judgement_block_count for r in rows)} "
                  f"unlabelled-judgement={summary.report_unlabelled_judgement_count}")
        print(f"[value-sprint] wrote {args.out_md}")
        print(f"[value-sprint] wrote {args.out_jsonl}")
        if args.emit_proposals:
            added = vsh.emit_memory_proposals(
                rows, queue_path=args.proposals_queue)
            print(f"[value-sprint] emitted {len(added)} memory proposal(s) "
                  f"(PENDING — not written) to {args.proposals_queue}")
        return 0

    # report: re-render Markdown from a (possibly operator-edited) JSONL.
    rows = vsh.load_rows(args.jsonl)
    summary = vsh.summarize(rows)
    backend_label = rows[0].retrieval_backend if rows else "deterministic"
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(
        vsh.render_markdown(rows, summary, backend_label=backend_label),
        encoding="utf-8")
    print(f"[value-sprint] re-rendered {args.out_md} from {args.jsonl}")
    return 0


# ---------- v3.0 retrieval evaluation CLI ----------

_RETRIEVAL_EVAL_CASES = ROOT / "demos" / "retrieval_eval_cases.jsonl"
_RETRIEVAL_REPORT_PATH_CASES = ROOT / "demos" / "retrieval_report_path_cases.jsonl"
_RETRIEVAL_HARDCORPUS_CASES = ROOT / "demos" / "retrieval_hardcorpus_cases.jsonl"
_SOURCE_REGISTRY_DEFAULT = ROOT / "demos" / "source_registry.jsonl"
_PROPOSAL_QUEUE_DEFAULT = ROOT / "reports" / "source_proposal_review_queue.jsonl"
_MEMORY_CANDIDATES_DEFAULT = ROOT / "demos" / "memory_proposal_candidates.jsonl"

def _retrieval_eval_cli(argv: list[str]) -> int:
    """Run the v3.0 retrieval evaluation harness (read-only, measurement only).

    Builds the named pack into a throwaway registry, scores the eval cases
    against the frozen ``query_knowledge`` retrieval path, and prints a readable
    summary. Changes no retrieval, ranking, source-selection, composer, or
    memory behaviour; writes reports only when ``--out-md`` / ``--out-jsonl``
    are given.

    With ``--probe-report-path`` (v3.0.1) it instead traces each case through the
    full report path — raw candidates -> selected evidence -> final report
    citations — and reports the first stage at which a forbidden source/term
    appears (or "not reproduced"). Still entirely read-only.

    With ``--probe-hygiene`` (v3.0.2) it runs the harder near-neighbour corpus
    and classifies every retrieved candidate into the wrong-source taxonomy
    (forbidden_bleed / on_topic_neighbour / ambiguous / expected_gap) plus
    chunk/source hygiene diagnostics. Still entirely read-only.
    """
    from agent import retrieval_eval_harness as reh

    parser = argparse.ArgumentParser(
        prog="workbench.py retrieval-eval",
        description="Measure retrieval quality for a pack (read-only). Exposes "
                    "the baseline; changes no retrieval/ranking/source logic.")
    parser.add_argument("--pack", default="m365_coding_assistant",
                        help="pack manifest directory under packs/")
    parser.add_argument("--backend", default="hybrid",
                        choices=["deterministic", "hybrid"],
                        help="knowledge retrieval backend (default hybrid — the "
                             "meaningful lexical path; deterministic recall is "
                             "near-chance and shown only as a floor)")
    parser.add_argument("--cases", default=None,
                        help="path to the case JSONL (defaults to the retrieval "
                             "eval cases, or the report-path cases under "
                             "--probe-report-path)")
    parser.add_argument("--probe-report-path", action="store_true",
                        help="trace the full report path and localise the stage "
                             "where relevance bleed is introduced (read-only)")
    parser.add_argument("--probe-hygiene", action="store_true",
                        help="run the harder near-neighbour corpus and classify "
                             "every candidate into the wrong-source taxonomy "
                             "with chunk/source hygiene diagnostics (read-only)")
    parser.add_argument("--out-md", default=None,
                        help="optional path to write a Markdown report")
    parser.add_argument("--out-jsonl", default=None,
                        help="optional path to write a JSONL report")
    args = parser.parse_args(argv)

    # seed=False: no memory writes; this exercises the knowledge path only.
    service = _build_sprint_service(args.pack, args.backend, seed=False)

    if args.probe_hygiene:
        cases_path = args.cases or str(_RETRIEVAL_HARDCORPUS_CASES)
        cases = reh.load_cases(cases_path)
        results = reh.run_hygiene(service, cases)
        summary = reh.summarize_hygiene(results)
        print(reh.render_hygiene_markdown(
            results, summary, pack_label=args.pack,
            backend_label=args.backend))
        if args.out_md or args.out_jsonl:
            reh.write_hygiene_reports(
                results, summary,
                md_path=args.out_md or (ROOT / "reports"
                                        / "hardcorpus_hygiene_latest.md"),
                jsonl_path=args.out_jsonl or (ROOT / "reports"
                                              / "hardcorpus_hygiene_latest.jsonl"),
                pack_label=args.pack, backend_label=args.backend)
            if args.out_md:
                print(f"[hygiene-probe] wrote {args.out_md}")
            if args.out_jsonl:
                print(f"[hygiene-probe] wrote {args.out_jsonl}")
        return 0

    if args.probe_report_path:
        cases_path = args.cases or str(_RETRIEVAL_REPORT_PATH_CASES)
        cases = reh.load_cases(cases_path)
        results = reh.run_probe(service, cases)
        summary = reh.summarize_probe(results)
        print(reh.render_probe_markdown(
            results, summary, pack_label=args.pack,
            backend_label=args.backend))
        if args.out_md or args.out_jsonl:
            reh.write_probe_reports(
                results, summary,
                md_path=args.out_md or (ROOT / "reports"
                                        / "report_path_probe_latest.md"),
                jsonl_path=args.out_jsonl or (ROOT / "reports"
                                              / "report_path_probe_latest.jsonl"),
                pack_label=args.pack, backend_label=args.backend)
            if args.out_md:
                print(f"[report-path-probe] wrote {args.out_md}")
            if args.out_jsonl:
                print(f"[report-path-probe] wrote {args.out_jsonl}")
        return 0

    cases = reh.load_cases(args.cases or str(_RETRIEVAL_EVAL_CASES))
    results = reh.run_eval(service, cases)
    summary = reh.summarize(results)

    print(reh.render_markdown(
        results, summary, pack_label=args.pack, backend_label=args.backend))
    if args.out_md or args.out_jsonl:
        reh.write_reports(
            results, summary,
            md_path=args.out_md or (ROOT / "reports"
                                    / "retrieval_eval_latest.md"),
            jsonl_path=args.out_jsonl or (ROOT / "reports"
                                          / "retrieval_eval_latest.jsonl"),
            pack_label=args.pack, backend_label=args.backend)
        if args.out_md:
            print(f"[retrieval-eval] wrote {args.out_md}")
        if args.out_jsonl:
            print(f"[retrieval-eval] wrote {args.out_jsonl}")
    return 0


# ---------- v4.0 source registry CLI (read-only metadata) ----------

def _source_registry_cli(argv: list[str]) -> int:
    """Inspect the source registry (read-only metadata; v4.0).

    ``list`` prints a deterministic table of every entry with its stored and
    *computed* effective freshness status; ``inspect <id>`` prints one entry's
    full metadata; ``audit`` prints a deterministic lifecycle/metadata risk
    report (v4.1); ``propose-updates`` turns that audit into deterministic
    source-maintenance proposals (v4.2) — printed to stdout, or written to
    ``--out`` as JSONL. All are pure reads of source metadata: the registry and
    source files are never modified, no memory ledger is written, and no
    retrieval, ranking, source-selection, grounding, or composer behaviour is
    touched. Proposals are tasks requiring human approval, never applied
    automatically; the registry annotates sources but never becomes the evidence,
    and neither the audit nor a proposal decides a source is false.
    """
    from agent import source_registry as sr

    if argv and argv[0] == "proposal-review":
        return _proposal_review_cli(argv[1:])

    parser = argparse.ArgumentParser(
        prog="workbench.py source-registry",
        description="Inspect source metadata (read-only). The registry annotates "
                    "sources; it changes no retrieval/ranking/grounding logic.")
    parser.add_argument(
        "action", choices=["list", "inspect", "audit", "propose-updates"],
        help="list all entries, inspect a single source_id, audit "
             "lifecycle/metadata risk, or propose source-maintenance updates")
    parser.add_argument("source_id", nargs="?", default=None,
                        help="the source_id to inspect (required for 'inspect')")
    parser.add_argument("--registry", default=str(_SOURCE_REGISTRY_DEFAULT),
                        help="path to the source registry JSONL")
    parser.add_argument("--out", default=None,
                        help="for 'propose-updates': write proposals to this "
                             "JSONL file instead of printing them (the only file "
                             "this command may write)")
    args = parser.parse_args(argv)

    entries = sr.load_registry(args.registry)

    if args.action == "list":
        print(sr.render_registry_markdown(entries))
        return 0

    if args.action == "audit":
        report = sr.audit_registry(entries)
        print(sr.render_audit_markdown(report))
        return 0

    if args.action == "propose-updates":
        proposals = sr.propose_source_updates(entries)
        if args.out:
            sr.write_proposals(proposals, args.out)
            print(f"[source-registry] wrote {len(proposals)} proposal(s) to "
                  f"{args.out} (registry unchanged; not applied)")
        else:
            print(sr.render_proposals_markdown(proposals))
        return 0

    if not args.source_id:
        print("[source-registry] inspect requires a source_id", file=sys.stderr)
        return 2
    index = sr.index_by_id(entries)
    entry = index.get(args.source_id)
    if entry is None:
        print(f"[source-registry] no entry for source_id {args.source_id!r}",
              file=sys.stderr)
        return 1
    print(sr.render_entry_markdown(entry, entries))
    return 0


def _proposal_review_cli(argv: list[str]) -> int:
    """Triage v4.2 source-update proposals (review-state only; v4.3).

    ``import-proposals <proposals_jsonl>`` creates ``pending`` review records
    (idempotent: duplicate proposal_ids are skipped); ``list`` prints a
    deterministic queue view; ``review <proposal_id> --status ...`` records an
    approve/reject/defer decision. Each command writes at most the one review
    queue file: the registry and source files are never modified, no memory
    ledger is written, and no retrieval/ranking/grounding/composer behaviour is
    touched. **Approved does not mean applied** — ``applied`` stays false in
    v4.3 and nothing is applied automatically.
    """
    from agent import source_registry as sr

    parser = argparse.ArgumentParser(
        prog="workbench.py source-registry proposal-review",
        description="Review source-update proposals (review-state only; "
                    "approved != applied; nothing is applied).")
    parser.add_argument(
        "subaction", choices=["list", "review", "import-proposals"],
        help="list the queue, review one proposal, or import proposals as "
             "pending review records")
    parser.add_argument(
        "target", nargs="?", default=None,
        help="proposal_id (for 'review') or proposals JSONL path (for "
             "'import-proposals')")
    parser.add_argument("--queue", default=str(_PROPOSAL_QUEUE_DEFAULT),
                        help="path to the review queue JSONL (the only file "
                             "these commands may write)")
    parser.add_argument("--status",
                        choices=["approved", "rejected", "deferred"],
                        default=None, help="for 'review': the decision to record")
    parser.add_argument("--note", default=None,
                        help="for 'review': an optional review note")
    parser.add_argument("--reviewer", default=None,
                        help="for 'review': who recorded the decision")
    parser.add_argument("--reviewed-at", default=None, dest="reviewed_at",
                        help="for 'review': an explicit ISO review timestamp")
    args = parser.parse_args(argv)

    queue = sr.load_review_queue(args.queue)

    if args.subaction == "list":
        print(sr.render_review_queue_markdown(queue))
        return 0

    if args.subaction == "import-proposals":
        if not args.target:
            print("[source-registry] import-proposals requires a proposals "
                  "JSONL path", file=sys.stderr)
            return 2
        proposals = sr.load_proposal_dicts(args.target)
        merged = sr.import_proposals_to_queue(proposals, queue)
        new_count = len(merged) - len(queue)
        sr.save_review_queue(merged, args.queue)
        print(f"[source-registry] imported {len(proposals)} proposal(s) into "
              f"{args.queue} ({new_count} new; registry unchanged; nothing "
              f"applied)")
        return 0

    # review
    if not args.target:
        print("[source-registry] review requires a proposal_id", file=sys.stderr)
        return 2
    if not args.status:
        print("[source-registry] review requires --status "
              "approved|rejected|deferred", file=sys.stderr)
        return 2
    try:
        updated = sr.apply_review_to_queue(
            queue, args.target, args.status, reviewer=args.reviewer,
            note=args.note, reviewed_at=args.reviewed_at)
    except ValueError as exc:
        print(f"[source-registry] {exc}", file=sys.stderr)
        return 1
    sr.save_review_queue(updated, args.queue)
    print(f"[source-registry] proposal-review {args.target} -> {args.status} "
          f"(queue updated; registry unchanged; not applied)")
    return 0


def _memory_proposals_cli(argv: list[str]) -> int:
    """Build typed, evidence-bound memory proposals for review (v5.0).

    ``list`` prints a deterministic Markdown review of the candidates (typed,
    with an approvable flag and any invalid reason); ``build`` builds proposals
    and either prints them as Markdown or, with ``--out``, writes them as JSONL.
    This is proposal-quality only: candidates are classified and scored but no
    memory is ever written. Every proposal requires human approval, and
    vague/operator-task/unsupported candidates are surfaced as
    ``INVALID_CANDIDATE`` (never approvable, never silently dropped). The memory
    ledger is never touched, and no retrieval/ranking/grounding/composer/source
    behaviour is changed.
    """
    from agent import memory_proposal_quality as mpq

    parser = argparse.ArgumentParser(
        prog="workbench.py memory-proposals",
        description="Build typed, evidence-bound memory proposals for human "
                    "review (proposal-quality only; no memory is written).")
    parser.add_argument(
        "action", choices=["list", "build"],
        help="list the typed proposals as Markdown, or build them (Markdown by "
             "default, or JSONL with --out)")
    parser.add_argument(
        "candidates", nargs="?", default=str(_MEMORY_CANDIDATES_DEFAULT),
        help="path to the raw memory-candidate JSONL")
    parser.add_argument("--out", default=None,
                        help="for 'build': write proposals to this JSONL file "
                             "instead of printing them (the only file this "
                             "command may write; never the memory ledger)")
    args = parser.parse_args(argv)

    candidates = mpq.load_memory_candidates(args.candidates)
    proposals = mpq.build_memory_proposals(candidates)

    if args.action == "build" and args.out:
        mpq.write_memory_proposals(proposals, args.out)
        approvable = sum(1 for p in proposals if mpq.is_approvable_as_memory(p))
        print(f"[memory-proposals] wrote {len(proposals)} proposal(s) to "
              f"{args.out} ({approvable} approvable as memory; no memory "
              f"written; all require human approval)")
        return 0

    print(mpq.render_memory_proposals_markdown(proposals))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "value-sprint":
        return _value_sprint_cli(argv[1:])
    if argv and argv[0] == "retrieval-eval":
        return _retrieval_eval_cli(argv[1:])
    if argv and argv[0] == "source-registry":
        return _source_registry_cli(argv[1:])
    if argv and argv[0] == "memory-proposals":
        return _memory_proposals_cli(argv[1:])
    parser = argparse.ArgumentParser(description="Concept Memory Workbench v1.1")
    parser.add_argument("--ledger", default=str(_DEFAULT_LEDGER),
                        help="path to the JSONL memory ledger")
    parser.add_argument("--seed", dest="seed", action="store_true",
                        default=True, help="seed the project memories (default)")
    parser.add_argument("--no-seed", dest="seed", action="store_false",
                        help="start with an empty workbench")
    parser.add_argument("--demo", action="store_true",
                        help="run the Friday -> Monday example and exit")
    parser.add_argument("--pack-root", default=str(_DEFAULT_PACK_ROOT),
                        help="directory holding project packs")
    parser.add_argument("--pack", default=None,
                        help="name or id of the project pack to open")
    args = parser.parse_args(argv)

    if args.demo:
        # Self-contained example on a fresh bank; no seeding, no ledger writes.
        _run_demo()
        return 0

    registry = PackRegistry(args.pack_root)

    if args.pack:
        pack = registry.get_pack(args.pack)
        if pack is None:
            pack = registry.create_pack(args.pack)
        registry.set_active_pack(pack.pack_id)
        service = WorkbenchService.from_pack(pack, registry=registry)
        if args.seed:
            _seed_pack_if_empty(service, pack)
        _repl(service, registry)
        return 0

    service = WorkbenchService(
        ledger_path=args.ledger,
        fresh=True,
        queue_path=str(Path(args.ledger).with_name("workbench_proposals.jsonl")),
        knowledge_path=str(Path(args.ledger).with_name("workbench_knowledge.jsonl")),
        registry=registry,
    )
    if args.seed and _SEED_FILE.exists():
        service.seed_from(_SEED_FILE)

    _repl(service, registry)
    return 0


# ---------- optional Streamlit UI (only when launched via `streamlit run`) ----

def _under_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


def _render_streamlit() -> None:  # pragma: no cover - requires streamlit
    import streamlit as st

    st.set_page_config(page_title="Concept Memory Workbench v1.1")
    st.title("Concept Memory Workbench v1.1")
    st.caption("Offline workbench over the frozen v1.0 memory. "
               "Retrieval is not grounding; the verifier decides.")

    if "service" not in st.session_state:
        svc = WorkbenchService(ledger_path=str(_DEFAULT_LEDGER), fresh=True)
        if _SEED_FILE.exists():
            svc.seed_from(_SEED_FILE)
        st.session_state.service = svc
    service: WorkbenchService = st.session_state.service

    with st.expander("Add memory"):
        text = st.text_input("Memory text", key="add_text")
        tags = st.text_input("Tags (comma-separated)", key="add_tags")
        if st.button("Add") and text.strip():
            tag_list = [t.strip() for t in tags.split(",") if t.strip()]
            entry = service.add_memory(text.strip(), source="workbench",
                                       tags=tag_list)
            st.success(f"Stored {entry.memory_id}")

    st.subheader("Query memory")
    query = st.text_input("Query text", key="query_text")
    if st.button("Query") and query.strip():
        audit = service.query_memory(query.strip())
        st.write({
            "candidate_retrieved": audit.candidate_retrieved,
            "verifier_verdict": audit.verifier_verdict,
            "memory_used": audit.memory_used,
            "refused": audit.refused,
            "cited_memory_ids": audit.cited_memory_ids,
        })
        st.code("\n".join(
            f"{c.memory_id} (rank {c.rank}, act {c.activation:.4f}) "
            f"-> {c.verdict.upper()}" for c in audit.candidates
        ) or "(no candidates)")
        st.text(audit.response_text)

    st.subheader("Memory ledger")
    st.dataframe(service.export_ledger())

    if st.button("Delete first active memory"):
        active = service.ledger.active_entries()
        if active:
            mid = active[0].memory_id
            service.delete_memory(mid)
            st.warning(f"Deleted {mid}; it can no longer be cited.")


if _under_streamlit():  # pragma: no cover - requires streamlit
    _render_streamlit()
elif __name__ == "__main__":
    raise SystemExit(main())
