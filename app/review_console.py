"""Review Console (v1.7): a local operator UI over the frozen workbench.

A small, fully offline console for *operating* project packs: pick the active
pack, review the proposal backlog, resolve conflicts, inspect memories and
knowledge, run audited queries, and export a pack. It adds no memory logic — all
behaviour comes from :class:`ReviewService`, a thin layer over the untouched
:class:`WorkbenchService` (retrieve -> verify -> ground). The console can never
bypass an approval gate or the grounding policy; it only surfaces them.

Run as a CLI (default, no extra dependencies):

    python app/review_console.py --pack concept-cells   # operate a pack
    python app/review_console.py                         # global stores

Or, if Streamlit is installed, as a local web app:

    streamlit run app/review_console.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent.project_packs import PackRegistry  # noqa: E402
from agent.review_service import ReviewService  # noqa: E402
from agent.workbench_service import WorkbenchService  # noqa: E402

_DEFAULT_PACK_ROOT = ROOT / "demos" / "packs"
_DEFAULT_LEDGER = ROOT / "demos" / "review_ledger.jsonl"

_HELP = """\
Commands:
  dashboard                 show active pack + review backlog counts
  packs                     list project packs (active marked *)
  pack-use <name>           switch the active pack
  pack-info                 show the active pack's paths and counts
  proposals                 list pending candidate memories
  conflicts                 list flagged conflicts and duplicates
  approve <id>              approve a proposal (refused for hard dup/conflict)
  approve-new <id>          approve as a new memory (explicit override)
  supersede <id> <old_id>   approve, superseding an existing memory
  reject <id>               reject a proposal
  edit <id> <text>          edit a proposal's text
  write-approved            write all approved proposals into the bank
  dispute <mem_id> [other]  flag a live memory as disputed
  memories                  list memories by status (active/deleted/...)
  knowledge                 list imported knowledge sources
  query <text>              audited query (memory + knowledge, kept separate)
  export <path>             export the active pack to a .zip bundle
  help                      show this help
  quit / exit               leave the console
"""


# ---------- shared rendering ----------

def _print_dashboard(review: ReviewService) -> None:
    state = review.get_dashboard_state()
    pack = state["active_pack"]
    if pack is None:
        print("  active pack: (none — using global stores)")
    else:
        print(f"  active pack: {pack['name']} ({pack['pack_id']})")
    print(f"  pending proposals : {state['pending_proposals']}")
    print(f"  conflicts         : {state['conflicts']}")
    print(f"  duplicates        : {state['duplicates']}")
    counts = state["memory_counts"]
    print("  memories          : "
          + ", ".join(f"{k}={v}" for k, v in counts.items()))
    print(f"  knowledge sources : {state['knowledge_sources']}")
    print(f"  knowledge backend : {state['knowledge_backend']}")


def _print_proposals(rows: list[dict], empty_msg: str) -> None:
    if not rows:
        print(f"  {empty_msg}")
        return
    for p in rows:
        tags = ",".join(p.get("tags") or [])
        line = (f"  {p['proposal_id']}  [{p['kind']}] "
                f"(conf {p['confidence']:.2f})  {p['canonical_text']}")
        if tags:
            line += f"   (tags: {tags})"
        print(line)
        verdict = p.get("lifecycle_verdict")
        if verdict and verdict != "new":
            against = (f" vs {p['lifecycle_candidate_id']}"
                       if p.get("lifecycle_candidate_id") else "")
            print(f"      lifecycle: {verdict.upper()}{against} "
                  f"— {p.get('lifecycle_reason') or ''}")


def _print_memories(review: ReviewService) -> None:
    table = review.get_memory_table()
    if not any(table.values()):
        print("  (no memories yet)")
        return
    for status in ("active", "disputed", "superseded", "deleted"):
        rows = table.get(status, [])
        if not rows:
            continue
        print(f"  {status} ({len(rows)}):")
        for row in rows:
            extra = ""
            if row.get("superseded_by"):
                extra = f"  -> {row['superseded_by']}"
            elif row.get("conflicts_with"):
                extra = f"  conflicts_with {', '.join(row['conflicts_with'])}"
            print(f"    {row['memory_id']}  {row['canonical_text']}{extra}")


def _print_knowledge(review: ReviewService) -> None:
    rows = review.get_knowledge_table()
    if not rows:
        print("  (no imported knowledge sources)")
        return
    for src in rows:
        version = f" v{src['version']}" if src.get("version") else ""
        print(f"  {src['source_id']}  {src['source_name']}{version}  "
              f"[{src['domain']}/{src['authority']}]  "
              f"({src['chunks']} chunks)")


def _print_query_audit(audit: dict) -> None:
    print(f'  query           = "{audit["query"]}"')
    print(f"  route           = {audit['route']}")
    print(f"  memory_used     = {str(audit['memory_used']).lower()}")
    print(f"  knowledge_used  = {str(audit['knowledge_used']).lower()}")
    print(f"  model_prior     = {str(audit['model_prior_used']).lower()}")
    print(f"  backend_name    = {audit['backend_name']}")
    memory = audit.get("memory")
    if memory:
        print(f"  verifier        = {memory['verifier_verdict'].upper()}")
        if memory.get("cited_memory_ids"):
            print(f"  cited memory    = {', '.join(memory['cited_memory_ids'])}")
    knowledge = audit.get("knowledge")
    if knowledge and knowledge.get("cited_source_ids"):
        print(f"  cited knowledge = {', '.join(knowledge['cited_source_ids'])}")
    for caution in audit.get("cautions", []):
        print(f"  ! {caution}")


# ---------- CLI ----------

def _repl(review: ReviewService, registry: PackRegistry | None) -> None:
    print("Concept Memory Review Console v1.7 "
          "(offline; type 'help', 'quit' to exit)")
    _print_dashboard(review)
    while True:
        try:
            raw = input("review> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not raw:
            continue
        cmd, _, arg = raw.partition(" ")
        cmd = cmd.lower()
        arg = arg.strip()

        if cmd in {"quit", "exit"}:
            return
        elif cmd == "help":
            print(_HELP)
        elif cmd == "dashboard":
            _print_dashboard(review)
        elif cmd in {"packs", "pack-list"}:
            _print_packs(registry)
        elif cmd == "pack-use":
            if registry is None:
                print("  packs are disabled; relaunch with --pack-root <dir>")
            elif not arg:
                print("  usage: pack-use <name>")
            else:
                try:
                    pack = review.select_pack(arg)
                    print(f"  active pack is now {pack.name} ({pack.pack_id})")
                except (KeyError, ValueError) as exc:
                    print(f"  {exc}")
        elif cmd == "pack-info":
            info = review.service.active_pack_info()
            if info is None:
                print("  no active pack (using global stores)")
            else:
                _print_pack_info(info)
        elif cmd == "proposals":
            _print_proposals(review.list_pending_proposals(),
                             "no pending proposals")
        elif cmd == "conflicts":
            rows = review.list_conflicts() + review.list_duplicates()
            _print_proposals(rows, "no conflicts or duplicates")
        elif cmd == "approve":
            _do_review(review, arg, "approve")
        elif cmd == "approve-new":
            _do_review(review, arg, "approve_new")
        elif cmd == "supersede":
            pid, _, old_id = arg.partition(" ")
            if not pid or not old_id.strip():
                print("  usage: supersede <proposal_id> <old_memory_id>")
            else:
                res = review.review_proposal(
                    pid, "approve_superseding",
                    old_memory_id=old_id.strip())
                print(f"  supersede {pid}: {str(res['ok']).lower()}")
        elif cmd == "reject":
            _do_review(review, arg, "reject")
        elif cmd == "edit":
            pid, _, new_text = arg.partition(" ")
            if not pid or not new_text.strip():
                print("  usage: edit <proposal_id> <new text>")
            else:
                res = review.review_proposal(pid, "edit",
                                             new_text=new_text.strip())
                print(f"  edit {pid}: {str(res['ok']).lower()}")
        elif cmd == "write-approved":
            written = review.write_approved()
            print(f"  wrote {len(written)} memories: {', '.join(written) or '-'}")
        elif cmd == "dispute":
            mid, _, other = arg.partition(" ")
            if not mid:
                print("  usage: dispute <memory_id> [conflict_id]")
            else:
                ok = review.mark_memory_disputed(mid, other.strip() or None)
                print(f"  dispute {mid}: {str(ok).lower()}")
        elif cmd == "memories":
            _print_memories(review)
        elif cmd == "knowledge":
            _print_knowledge(review)
        elif cmd == "query":
            if not arg:
                print("  usage: query <text>")
            else:
                _print_query_audit(review.run_audited_query(arg))
        elif cmd == "export":
            if not arg:
                print("  usage: export <path>")
            else:
                try:
                    out = review.export_active_pack(arg)
                    print(f"  exported active pack to {out}")
                except (ValueError, KeyError, OSError) as exc:
                    print(f"  {exc}")
        else:
            print(f"  unknown command: {cmd} (type 'help')")


def _do_review(review: ReviewService, proposal_id: str, action: str) -> None:
    if not proposal_id:
        print(f"  usage: {action} <proposal_id>")
        return
    res = review.review_proposal(proposal_id, action)
    print(f"  {action} {proposal_id}: {str(res['ok']).lower()}")
    if not res["ok"] and action == "approve":
        print("    (refused — a hard duplicate/conflict needs "
              "approve-new or supersede)")


def _print_packs(registry: PackRegistry | None) -> None:
    if registry is None:
        print("  packs are disabled; relaunch with --pack-root <dir>")
        return
    packs = registry.list_packs()
    if not packs:
        print("  (no packs yet)")
        return
    active = registry.get_active_pack()
    active_id = active.pack_id if active else None
    for pack in packs:
        marker = "*" if pack.pack_id == active_id else " "
        print(f"  {marker} {pack.pack_id}  ({pack.name})")


def _print_pack_info(info: dict) -> None:
    print(f"  pack_id   : {info['pack_id']}")
    print(f"  name      : {info['name']}")
    print(f"  root      : {info['root_path']}")
    print(f"  memories  : {info['memory_entries']} ledger entries")
    print(f"  proposals : {info['proposal_entries']} queued")
    print(f"  knowledge : {info['knowledge_records']} records")


def _seed_pack_if_empty(service: WorkbenchService, pack) -> None:
    """Seed a pack from its manifest seed settings if it has no content yet.

    This is app-layer glue only — it reuses the workbench's existing
    ``seed_from`` / ``import_knowledge`` methods and adds no memory logic, so a
    freshly opened demo pack has something to inspect.
    """
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


def _build_review(args) -> tuple[ReviewService, PackRegistry]:
    """Build a ReviewService from CLI args (pack-bound or global)."""
    registry = PackRegistry(args.pack_root)
    if args.pack:
        pack = registry.get_pack(args.pack)
        if pack is None:
            pack = registry.create_pack(args.pack)
        registry.set_active_pack(pack.pack_id)
        service = WorkbenchService.from_pack(pack, registry=registry)
        _seed_pack_if_empty(service, pack)
    else:
        service = WorkbenchService(
            ledger_path=args.ledger,
            fresh=True,
            queue_path=str(Path(args.ledger).with_name("review_proposals.jsonl")),
            knowledge_path=str(Path(args.ledger).with_name("review_knowledge.jsonl")),
            registry=registry,
        )
    return ReviewService(service, registry), registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Concept Memory Review Console v1.7")
    parser.add_argument("--pack-root", default=str(_DEFAULT_PACK_ROOT),
                        help="directory holding project packs")
    parser.add_argument("--pack", default=None,
                        help="name or id of the project pack to operate")
    parser.add_argument("--ledger", default=str(_DEFAULT_LEDGER),
                        help="ledger path when no pack is selected")
    args = parser.parse_args(argv)

    review, registry = _build_review(args)
    _repl(review, registry)
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

    st.set_page_config(page_title="Concept Memory Review Console v1.7")
    st.title("Concept Memory Review Console v1.7")
    st.caption("Offline operator console over the frozen v1.0 memory. "
               "Approval gates and grounding are enforced, never bypassed.")

    registry = PackRegistry(_DEFAULT_PACK_ROOT)

    if "review" not in st.session_state:
        packs = registry.list_packs()
        if packs:
            active = registry.get_active_pack() or packs[0]
            registry.set_active_pack(active.pack_id)
            service = WorkbenchService.from_pack(active, registry=registry)
        else:
            service = WorkbenchService(
                ledger_path=str(_DEFAULT_LEDGER), fresh=True, registry=registry)
        st.session_state.review = ReviewService(service, registry)
    review: ReviewService = st.session_state.review

    # Pack selector.
    packs = registry.list_packs()
    if packs:
        names = [p.name for p in packs]
        active = registry.get_active_pack()
        idx = names.index(active.name) if active and active.name in names else 0
        chosen = st.selectbox("Active pack", names, index=idx)
        if active is None or chosen != active.name:
            review.select_pack(chosen)
            st.rerun()

    state = review.get_dashboard_state()
    cols = st.columns(4)
    cols[0].metric("Pending", state["pending_proposals"])
    cols[1].metric("Conflicts", state["conflicts"])
    cols[2].metric("Duplicates", state["duplicates"])
    cols[3].metric("Knowledge", state["knowledge_sources"])
    st.caption(f"Knowledge backend: {state['knowledge_backend']}")

    st.subheader("Pending proposals")
    pending = review.list_pending_proposals()
    if pending:
        st.dataframe(pending)
        pid = st.selectbox(
            "Proposal", [p["proposal_id"] for p in pending], key="pid")
        action = st.radio(
            "Action", ["approve", "approve_new", "reject"], horizontal=True)
        if st.button("Apply"):
            res = review.review_proposal(pid, action)
            (st.success if res["ok"] else st.warning)(
                f"{action} {pid}: ok={res['ok']}")
        if st.button("Write approved"):
            written = review.write_approved()
            st.success(f"Wrote {len(written)} memories")
    else:
        st.write("No pending proposals.")

    st.subheader("Memories")
    st.write(review.get_memory_table())

    st.subheader("Knowledge sources")
    st.dataframe(review.get_knowledge_table())

    st.subheader("Audited query")
    query = st.text_input("Query text", key="query_text")
    if st.button("Run query") and query.strip():
        st.write(review.run_audited_query(query.strip()))

    if state["active_pack"] is not None:
        st.subheader("Export pack")
        out_path = st.text_input("Bundle path", value="pack_bundle.zip")
        if st.button("Export") and out_path.strip():
            out = review.export_active_pack(out_path.strip())
            st.success(f"Exported to {out}")


if _under_streamlit():  # pragma: no cover - requires streamlit
    _render_streamlit()
elif __name__ == "__main__":
    raise SystemExit(main())
