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
  backend               show the active knowledge retrieval backend
  packs                 list project packs (isolated workspaces)
  pack-create <name>    create a new project pack
  pack-use <name>       switch the active pack (re-points all stores)
  pack-info [name]      show the active pack (or a named pack) and its paths
  pack-export <name> <path>  export a pack to a .zip bundle
  pack-import <path>    import a pack bundle and register it
  pack-build <spec>     build a curated knowledge pack from a JSON/YAML spec
  pack-eval <pack> <eval-file>  validate a pack's retrieval with eval questions
  pack-report <pack>    show a pack's curated sources and chunk counts
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
        elif cmd in {"pack-use", "pack_use"}:
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


def main(argv: list[str] | None = None) -> int:
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
