"""Phase 3 — governed memory mutation orchestrator.

Wraps :mod:`src.agent.bank_admin` with a typed ``plan → apply`` flow:

1. ``plan(intent, …)`` runs all preflight checks and returns a frozen
   :class:`MemoryPlan` describing exactly what would happen. **No
   mutation occurs.** This is the default mode; the CLI exposes it as
   ``memory.py plan "instruction"``.
2. ``apply(plan, …)`` executes the plan against the overlay. It refuses
   to run if ``plan.blocked`` is true. On success it returns a
   :class:`MemoryResult` carrying the post-mutation validator report.

The orchestrator never imports the answer pipeline. It does not load
the encoder or the bank itself — those are supplied by the caller, so
this module remains importable in minimal test environments. The
production CLI (``scripts/memory.py``) handles the heavyweight loads.

Cardinal rule (mirrors Phase 2): the orchestrator must not mutate the
overlay unless the caller has explicitly produced a non-blocked plan
and called ``apply``. ``plan`` is read-only end-to-end.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from src.agent.bank_admin import (
    OverlayStore,
    OverlayStoreError,
    add_cell_from_text,
)
from src.agent.memory_intent_classifier import MemoryIntent, MemoryIntentKind
from src.agent.post_mutation_validator import (
    ValidatorReport,
    validate_add,
    validate_inspect,
    validate_remove,
)


@dataclass(frozen=True)
class MemoryPlan:
    """What ``apply`` would do.

    ``blocked`` is the safety gate. When true, ``apply`` must refuse to
    run. ``preflight_findings`` is a human-readable log of every check
    the planner performed so the CLI can show its working.
    """

    intent: MemoryIntent
    op: str
    op_kwargs: dict = field(default_factory=dict)
    preflight_findings: List[str] = field(default_factory=list)
    blocked: bool = False
    block_reason: str = ""

    def summary(self) -> str:
        head = f"PLAN op={self.op}"
        if self.blocked:
            return f"{head}  BLOCKED: {self.block_reason}"
        if self.op == "add":
            text = self.op_kwargs.get("text", "")
            preview = text[:60] + ("…" if len(text) > 60 else "")
            return f"{head}  add cell from text {preview!r}"
        if self.op == "remove":
            cid = self.op_kwargs.get("cell_id")
            reason = self.op_kwargs.get("reason", "")
            return f"{head}  tombstone cell {cid}: {reason!r}"
        if self.op == "inspect":
            return f"{head}  list provenance"
        return f"{head}  unknown"


@dataclass(frozen=True)
class MemoryResult:
    """Outcome of an applied plan."""

    plan: MemoryPlan
    applied: bool
    new_cell_id: Optional[int] = None
    tombstoned_cell_id: Optional[int] = None
    provenance: Optional[List[dict]] = None
    error: str = ""
    validator: Optional[ValidatorReport] = None

    def as_dict(self) -> dict:
        return {
            "applied": self.applied,
            "op": self.plan.op,
            "new_cell_id": self.new_cell_id,
            "tombstoned_cell_id": self.tombstoned_cell_id,
            "error": self.error,
            "validator": self.validator.as_dict() if self.validator else None,
            "preflight": list(self.plan.preflight_findings),
            "blocked": self.plan.blocked,
            "block_reason": self.plan.block_reason,
        }


# ---- planning ----

def _block(intent: MemoryIntent, op: str, findings: list, reason: str) -> MemoryPlan:
    findings.append(f"BLOCK: {reason}")
    return MemoryPlan(
        intent=intent,
        op=op,
        preflight_findings=list(findings),
        blocked=True,
        block_reason=reason,
    )


def plan(
    intent: MemoryIntent,
    *,
    overlay: OverlayStore,
    base_max_id: Optional[int] = None,
) -> MemoryPlan:
    """Run preflight checks and return a :class:`MemoryPlan`.

    ``base_max_id`` is required for ``add`` so the allocation guard can
    be checked before any encoder load. For ``remove`` it is optional
    (the overlay handles base/overlay ids transparently).
    """
    findings: list = [f"intent.kind={intent.kind}"]

    if intent.kind == MemoryIntentKind.UNKNOWN:
        note = intent.note or "instruction not recognised"
        return _block(intent, "unknown", findings, note)

    if intent.kind == MemoryIntentKind.ADD:
        text = str(intent.fields.get("text", "")).strip()
        if not text:
            return _block(intent, "add", findings, "add intent had empty text")
        findings.append(f"text length = {len(text)} chars")
        if base_max_id is None:
            return _block(
                intent, "add", findings,
                "add planning needs base_max_id; pass --bank-path on the CLI",
            )
        findings.append(f"base_max_id = {base_max_id}")
        anchored_str = overlay.get_meta("allocate_id_base")
        if anchored_str is not None:
            anchored = int(anchored_str)
            if int(base_max_id) > anchored:
                return _block(
                    intent, "add", findings,
                    f"base_max_id={base_max_id} exceeds overlay anchor "
                    f"{anchored}; base bank has grown since overlay was created",
                )
            findings.append(f"overlay anchored at {anchored}")
        else:
            findings.append("overlay has no prior anchor (first add)")
        return MemoryPlan(
            intent=intent,
            op="add",
            op_kwargs={"text": text, "base_max_id": int(base_max_id)},
            preflight_findings=list(findings),
        )

    if intent.kind == MemoryIntentKind.REMOVE:
        cell_id = intent.fields.get("cell_id")
        reason = str(intent.fields.get("reason", "")).strip()
        if not isinstance(cell_id, int):
            return _block(intent, "remove", findings, "missing cell_id")
        if not reason:
            return _block(intent, "remove", findings, "remove intent had empty reason")
        findings.append(f"cell_id = {cell_id}, reason length = {len(reason)}")
        # Check whether the cell is already tombstoned. Not a block — the
        # underlying API is idempotent — but the operator should know.
        existing = overlay._conn.execute(  # type: ignore[attr-defined]
            "SELECT reason FROM tombstones WHERE base_cell_id = ?",
            (int(cell_id),),
        ).fetchone()
        if existing is not None:
            findings.append(
                f"cell {cell_id} is already tombstoned "
                f"(prior reason: {str(existing[0])!r}); "
                f"apply will overwrite the reason"
            )
        return MemoryPlan(
            intent=intent,
            op="remove",
            op_kwargs={"cell_id": int(cell_id), "reason": reason},
            preflight_findings=list(findings),
        )

    if intent.kind == MemoryIntentKind.INSPECT:
        return MemoryPlan(
            intent=intent,
            op="inspect",
            op_kwargs={},
            preflight_findings=list(findings),
        )

    return _block(intent, "unknown", findings, f"unhandled intent kind {intent.kind!r}")


# ---- application ----

def apply(
    plan_obj: MemoryPlan,
    *,
    overlay: OverlayStore,
    encoder: Any = None,
    whiten_fn: Optional[Callable[[Any], Any]] = None,
    bank_dim: Optional[int] = None,
    source: Optional[str] = None,
    label: Optional[str] = None,
    theta: float = 0.30,
    validate: bool = True,
) -> MemoryResult:
    """Execute a non-blocked :class:`MemoryPlan` against ``overlay``.

    The ``encoder`` / ``whiten_fn`` / ``bank_dim`` kwargs are only
    required for ``add`` plans. For ``remove`` and ``inspect`` they are
    ignored and may be omitted.
    """
    if plan_obj.blocked:
        return MemoryResult(
            plan=plan_obj, applied=False,
            error=f"plan blocked: {plan_obj.block_reason}",
        )

    if plan_obj.op == "add":
        if encoder is None or whiten_fn is None or bank_dim is None:
            return MemoryResult(
                plan=plan_obj, applied=False,
                error="add apply requires encoder, whiten_fn, and bank_dim",
            )
        text = plan_obj.op_kwargs["text"]
        base_max_id = plan_obj.op_kwargs["base_max_id"]
        try:
            new_id, _ = add_cell_from_text(
                overlay,
                bank_dim=bank_dim,
                base_max_id=base_max_id,
                encoder=encoder,
                whiten_fn=whiten_fn,
                text=text,
                source=source,
                label=label,
                theta=theta,
            )
        except (OverlayStoreError, ValueError) as exc:
            return MemoryResult(
                plan=plan_obj, applied=False, error=f"{type(exc).__name__}: {exc}",
            )
        validator = validate_add(
            overlay, new_cell_id=new_id, expected_text=text,
        ) if validate else None
        return MemoryResult(
            plan=plan_obj, applied=True,
            new_cell_id=new_id, validator=validator,
        )

    if plan_obj.op == "remove":
        cell_id = plan_obj.op_kwargs["cell_id"]
        reason = plan_obj.op_kwargs["reason"]
        try:
            overlay.remove_cell(int(cell_id), reason=reason)
        except (OverlayStoreError, ValueError) as exc:
            return MemoryResult(
                plan=plan_obj, applied=False, error=f"{type(exc).__name__}: {exc}",
            )
        validator = validate_remove(
            overlay, cell_id=int(cell_id), expected_reason=reason,
        ) if validate else None
        return MemoryResult(
            plan=plan_obj, applied=True,
            tombstoned_cell_id=int(cell_id), validator=validator,
        )

    if plan_obj.op == "inspect":
        try:
            log = overlay.provenance()
        except Exception as exc:  # noqa: BLE001
            return MemoryResult(
                plan=plan_obj, applied=False, error=repr(exc),
            )
        validator = validate_inspect(overlay) if validate else None
        return MemoryResult(
            plan=plan_obj, applied=True,
            provenance=log, validator=validator,
        )

    return MemoryResult(
        plan=plan_obj, applied=False, error=f"unknown op {plan_obj.op!r}",
    )


__all__ = ["MemoryPlan", "MemoryResult", "plan", "apply"]
