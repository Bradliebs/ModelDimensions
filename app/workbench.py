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

_DEFAULT_LEDGER = ROOT / "demos" / "workbench_ledger.jsonl"
_DEFAULT_QUEUE = ROOT / "demos" / "workbench_proposals.jsonl"
_SEED_FILE = ROOT / "demos" / "seed_concept_cells_project.jsonl"
_INGEST_NOTE = ROOT / "demos" / "seed_ingestion_note.md"

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
    lines.append("  response:")
    for row in audit.response_text.splitlines():
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


def _repl(service: WorkbenchService) -> None:
    print("Concept Memory Workbench (v1.1) — offline. Type 'help' for commands.")
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
            print(_format_audit(service.query_memory(arg)))
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
        elif cmd == "demo":
            _run_demo()
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
    args = parser.parse_args(argv)

    if args.demo:
        # Self-contained example on a fresh bank; no seeding, no ledger writes.
        _run_demo()
        return 0

    service = WorkbenchService(
        ledger_path=args.ledger,
        fresh=True,
        queue_path=str(Path(args.ledger).with_name("workbench_proposals.jsonl")),
    )
    if args.seed and _SEED_FILE.exists():
        service.seed_from(_SEED_FILE)

    _repl(service)
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
