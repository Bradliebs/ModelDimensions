"""Phase 3 — post-mutation validator.

After the orchestrator applies a memory mutation, this layer reads the
overlay back and confirms the mutation actually landed as intended. It
exists because the V1 overlay path has historically been operated via
``python -c`` one-liners that silently swallow exceptions in shells
(PowerShell ``2>&1`` redirection, copy-paste truncation). A separate,
read-only post-check converts "the command returned 0" into "the row
exists with the expected text".

Scope is deliberately narrow:

* For an **add**, confirm a single new ``overlay_cells`` row exists at
  ``new_cell_id`` and its ``source_text`` matches the intent verbatim.
* For a **remove**, confirm a tombstone row exists for ``cell_id`` with
  the intent's reason.
* For an **inspect**, validation is trivially "did the read succeed".

The validator does NOT load the answer pipeline. A smoke ask through
Phi-3 takes ~40 s of cold load and is a separate, opt-in step.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from src.agent.bank_admin import OverlayStore


@dataclass(frozen=True)
class ValidatorReport:
    """Outcome of a post-mutation read-back.

    ``checks`` is the ordered audit trail; each entry is ``(name, ok,
    detail)``. ``ok`` is the conjunction of every check's ``ok``.
    """

    ok: bool
    op: str
    checks: List[Tuple[str, bool, str]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "op": self.op,
            "checks": [
                {"name": n, "ok": o, "detail": d} for n, o, d in self.checks
            ],
        }


def _fail(op: str, checks: list, name: str, detail: str) -> ValidatorReport:
    checks.append((name, False, detail))
    return ValidatorReport(ok=False, op=op, checks=list(checks))


def validate_add(
    overlay: OverlayStore,
    *,
    new_cell_id: int,
    expected_text: str,
) -> ValidatorReport:
    checks: list = []

    stored_text = overlay.fetch_overlay_text(new_cell_id)
    if stored_text is None:
        return _fail(
            "add", checks, "row_exists",
            f"no overlay_cells row at id={new_cell_id}",
        )
    checks.append(("row_exists", True, f"id={new_cell_id}"))

    if stored_text != expected_text:
        return _fail(
            "add", checks, "text_matches",
            f"stored text != intent text "
            f"(stored {stored_text!r}, expected {expected_text!r})",
        )
    checks.append(("text_matches", True, f"{len(expected_text)} chars"))

    # Provenance trail must record the add. Last entry should be ours.
    log = overlay.provenance()
    matching = [r for r in log if r["op"] == "add" and r["cell_id"] == new_cell_id]
    if not matching:
        return _fail(
            "add", checks, "provenance_logged",
            f"no provenance_log row op=add cell_id={new_cell_id}",
        )
    checks.append(("provenance_logged", True, f"{len(matching)} entry"))

    return ValidatorReport(ok=True, op="add", checks=checks)


def validate_remove(
    overlay: OverlayStore,
    *,
    cell_id: int,
    expected_reason: str,
) -> ValidatorReport:
    checks: list = []

    row = overlay._conn.execute(  # type: ignore[attr-defined]
        "SELECT reason, removed_at FROM tombstones WHERE base_cell_id = ?",
        (int(cell_id),),
    ).fetchone()
    if row is None:
        return _fail(
            "remove", checks, "tombstone_present",
            f"no tombstone row for cell_id={cell_id}",
        )
    checks.append(("tombstone_present", True, f"cell_id={cell_id}"))

    stored_reason = str(row[0])
    if stored_reason != expected_reason:
        return _fail(
            "remove", checks, "reason_matches",
            f"stored reason {stored_reason!r} != expected {expected_reason!r}",
        )
    checks.append(("reason_matches", True, f"{len(expected_reason)} chars"))

    log = overlay.provenance()
    matching = [
        r for r in log
        if r["op"] == "remove" and r["cell_id"] == cell_id
    ]
    if not matching:
        return _fail(
            "remove", checks, "provenance_logged",
            f"no provenance_log row op=remove cell_id={cell_id}",
        )
    checks.append(("provenance_logged", True, f"{len(matching)} entry"))

    return ValidatorReport(ok=True, op="remove", checks=checks)


def validate_inspect(overlay: OverlayStore) -> ValidatorReport:
    """Trivial validation: confirm provenance can be read at all."""
    try:
        log = overlay.provenance()
    except Exception as exc:  # noqa: BLE001
        return ValidatorReport(
            ok=False, op="inspect",
            checks=[("read_provenance", False, repr(exc))],
        )
    return ValidatorReport(
        ok=True, op="inspect",
        checks=[("read_provenance", True, f"{len(log)} entries")],
    )


__all__ = [
    "ValidatorReport",
    "validate_add",
    "validate_remove",
    "validate_inspect",
]
