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
_SEED_FILE = ROOT / "demos" / "seed_concept_cells_project.jsonl"

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

    service = WorkbenchService(ledger_path=args.ledger, fresh=True)
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
