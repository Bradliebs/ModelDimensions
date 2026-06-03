"""Review service (v1.7): a thin operator layer over :class:`WorkbenchService`.

This is the seam the review console (Streamlit or text mode) talks to. It owns
**no** memory logic of its own: it composes the existing, frozen
:class:`WorkbenchService` operations into the structured views and actions an
operator needs (dashboard, proposal review, conflict resolution, memory and
knowledge tables, audited query, pack export).

Every approval path goes through the workbench's own methods, so the verifier
REJECT override, the grounding policy, and the duplicate/conflict approval gates
are all enforced exactly as before. The review service can never approve a hard
duplicate or conflict via the plain ``approve`` action, and an audited query
returns the same :class:`CombinedAudit` the workbench produces.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from agent.memory_ledger import ACTIVE, DELETED, DISPUTED, SUPERSEDED
from agent.project_packs import PackRegistry, ProjectPack
from agent.workbench_service import WorkbenchService

# The proposal review verbs the console exposes, each mapped to an existing
# WorkbenchService method. "approve" is the gated path (refused for hard
# duplicates/conflicts); "approve_new" and "approve_superseding" are the
# explicit operator overrides the workbench already provides.
_PROPOSAL_ACTIONS = {
    "approve",
    "reject",
    "edit",
    "approve_new",
    "approve_superseding",
}

# Conflict/duplicate resolution verbs. These never silently bypass approval:
# they call the same explicit-override methods a human would use at the CLI.
_CONFLICT_RESOLUTIONS = {
    "approve_new",
    "approve_superseding",
    "reject",
    "dispute",
}

_MEMORY_STATUSES = (ACTIVE, DELETED, SUPERSEDED, DISPUTED)


class ReviewService:
    """Operator-facing orchestration over one :class:`WorkbenchService`.

    ``registry`` is optional; when absent it is taken from the wrapped service so
    pack export still works when the service was built with a registry.
    """

    def __init__(self, service: WorkbenchService,
                 registry: Optional[PackRegistry] = None):
        self.service = service
        self.registry = registry or getattr(service, "_registry", None)

    # -- dashboard --

    def get_dashboard_state(self) -> dict:
        """A single snapshot of the active pack and review backlog."""
        memory = self.get_memory_table()
        return {
            "active_pack": self.service.active_pack_info(),
            "pending_proposals": len(self.service.list_proposals("pending")),
            "conflicts": len(self.service.list_conflicts()),
            "duplicates": len(self.service.list_duplicates()),
            "memory_counts": {
                status: len(rows) for status, rows in memory.items()
            },
            "knowledge_sources": len(self.service.list_knowledge_sources()),
            "knowledge_backend": self.service.knowledge_backend_name(),
        }

    # -- proposals --

    def list_pending_proposals(self) -> List[dict]:
        """Pending candidate memories awaiting review, as plain dicts."""
        return [p.to_dict() for p in self.service.list_proposals("pending")]

    def list_conflicts(self) -> List[dict]:
        """Pending proposals flagged as a conflict (hard or possible)."""
        return [p.to_dict() for p in self.service.list_conflicts()]

    def list_duplicates(self) -> List[dict]:
        """Pending proposals flagged as a duplicate (hard or possible)."""
        return [p.to_dict() for p in self.service.list_duplicates()]

    def review_proposal(self, proposal_id: str, action: str, *,
                        new_text: Optional[str] = None,
                        old_memory_id: Optional[str] = None) -> dict:
        """Apply one review ``action`` to a proposal through the workbench.

        ``action`` is one of ``approve`` / ``reject`` / ``edit`` /
        ``approve_new`` / ``approve_superseding``. The plain ``approve`` is the
        gated path: the workbench refuses it for a hard duplicate or conflict, so
        the review service cannot bypass the approval rules. Returns a small
        result dict with ``ok`` reflecting the workbench's own outcome.
        """
        if action not in _PROPOSAL_ACTIONS:
            raise ValueError(f"unknown proposal action: {action!r}")

        if action == "approve":
            ok = self.service.approve_proposal(proposal_id)
        elif action == "reject":
            ok = self.service.reject_proposal(proposal_id)
        elif action == "edit":
            if new_text is None:
                raise ValueError("edit requires new_text")
            ok = self.service.edit_proposal(proposal_id, new_text)
        elif action == "approve_new":
            ok = self.service.approve_proposal_as_new(proposal_id)
        else:  # approve_superseding
            if old_memory_id is None:
                raise ValueError("approve_superseding requires old_memory_id")
            ok = self.service.approve_proposal_superseding(
                proposal_id, old_memory_id)

        return {"proposal_id": proposal_id, "action": action, "ok": bool(ok)}

    def review_conflict(self, proposal_id: str, resolution: str, *,
                        old_memory_id: Optional[str] = None,
                        disputed_memory_id: Optional[str] = None) -> dict:
        """Resolve a flagged conflict/duplicate without bypassing approval.

        ``resolution`` is one of ``approve_new`` (write it as its own memory),
        ``approve_superseding`` (replace ``old_memory_id``), ``reject`` (drop the
        proposal), or ``dispute`` (flag an existing live memory as contested,
        leaving both memories intact). Every branch delegates to an existing
        workbench method.
        """
        if resolution not in _CONFLICT_RESOLUTIONS:
            raise ValueError(f"unknown conflict resolution: {resolution!r}")

        if resolution == "approve_new":
            ok = self.service.approve_proposal_as_new(proposal_id)
        elif resolution == "approve_superseding":
            if old_memory_id is None:
                raise ValueError("approve_superseding requires old_memory_id")
            ok = self.service.approve_proposal_superseding(
                proposal_id, old_memory_id)
        elif resolution == "reject":
            ok = self.service.reject_proposal(proposal_id)
        else:  # dispute
            target = disputed_memory_id or old_memory_id
            if target is None:
                raise ValueError("dispute requires disputed_memory_id")
            ok = self.service.dispute_memory(target, conflict_id=None)

        return {
            "proposal_id": proposal_id,
            "resolution": resolution,
            "ok": bool(ok),
        }

    def write_approved(self) -> List[str]:
        """Write every approved, not-yet-written proposal; return new ids."""
        return [e.memory_id for e in self.service.write_approved_proposals()]

    # -- memory inspection --

    def mark_memory_disputed(self, memory_id: str,
                             conflict_id: Optional[str] = None) -> bool:
        """Flag a live memory as disputed (delegates to the workbench)."""
        return self.service.dispute_memory(memory_id, conflict_id)

    def get_memory_table(self) -> dict:
        """Group ledger entries by status: active/deleted/superseded/disputed."""
        table: dict[str, List[dict]] = {
            status: [] for status in _MEMORY_STATUSES
        }
        for row in self.service.export_ledger():
            status = row.get("status")
            if status in table:
                table[status].append(row)
        return table

    # -- knowledge inspection --

    def get_knowledge_table(self) -> List[dict]:
        """Active knowledge sources with their source/domain/authority metadata."""
        return self.service.list_knowledge_sources()

    # -- audited query --

    def run_audited_query(self, query_text: str) -> dict:
        """Run the combined query and return the full audit as a dict.

        The result is exactly the workbench's :class:`CombinedAudit`, so
        ``memory_used``, ``knowledge_used`` and the knowledge ``backend_name``
        are preserved. No grounding decision is made here.
        """
        audit = self.service.query_all(query_text).to_dict()
        # Surface the backend name at the top level too, mirroring the
        # knowledge audit's own field, so the console can show it directly.
        audit["backend_name"] = self.service.knowledge_backend_name()
        return audit

    # -- pack operations --

    def select_pack(self, pack_id_or_name: str) -> ProjectPack:
        """Switch the active pack (requires the service to have a registry)."""
        return self.service.switch_pack(pack_id_or_name)

    def export_active_pack(self, output_path: str | Path) -> Path:
        """Export the active pack to a ``.zip`` bundle via the registry."""
        info = self.service.active_pack_info()
        if info is None:
            raise ValueError("no active pack to export")
        if self.registry is None:
            raise ValueError("export requires a PackRegistry")
        return self.registry.export_pack(info["pack_id"], output_path)
