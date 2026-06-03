"""Response policy: the grounding guarantee.

This module is the trust boundary. Whatever the SLM says, the user only ever
receives text that satisfies these rules:

  * If no concept cell fired, the response states that memory is silent and
    cites nothing (``refused=True``).
  * If cells fired, the response is built from those fired memories and cites
    their ids (``memory_used=True``).
  * The SLM may never introduce a fact as memory-derived: ``enforce_grounding``
    drops any cited id that is not in the set of fired ids.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Sequence

from slm.schemas import GroundedResponse, MemoryQueryResult

if TYPE_CHECKING:  # avoid a runtime import cycle with orchestrator
    from .orchestrator import MemoryBank


def ground_from_cells(query_result: MemoryQueryResult,
                      bank: "MemoryBank") -> GroundedResponse:
    """Build an authoritative answer using ONLY the fired memories.

    The text is assembled from the canonical text of each fired memory. No
    language model is consulted, so there is no path for an unsupported fact
    to enter here.
    """
    if query_result.silent or not query_result.fired_memory_ids:
        return GroundedResponse(
            text="Memory is silent: no stored memory matched this query.",
            cited_memory_ids=[],
            memory_used=False,
            refused=True,
        )

    lines: List[str] = []
    cited: List[str] = []
    for mem_id in query_result.fired_memory_ids:
        rec = bank.get(mem_id)
        if rec is None:
            # Fired id with no backing record should be impossible; skip it
            # rather than fabricate content.
            continue
        lines.append(f"- ({mem_id}) {rec.canonical_text}")
        cited.append(mem_id)

    if not cited:
        return GroundedResponse(
            text="Memory is silent: no stored memory matched this query.",
            cited_memory_ids=[],
            memory_used=False,
            refused=True,
        )

    body = "\n".join(lines)
    return GroundedResponse(
        text=f"From memory:\n{body}",
        cited_memory_ids=cited,
        memory_used=True,
        refused=False,
    )


def enforce_grounding(slm_text: str,
                      fired_memory_ids: Sequence[str],
                      claimed_memory_ids: Sequence[str]) -> GroundedResponse:
    """Validate an SLM-phrased answer against the fired cells.

    The SLM is allowed to phrase the answer (``slm_text``) and to claim which
    memories it used (``claimed_memory_ids``). This function refuses outright
    if any claimed id was not actually fired — the SLM cannot smuggle in a
    memory that the geometry did not retrieve.
    """
    fired = set(fired_memory_ids)
    if not fired:
        return GroundedResponse(
            text="Memory is silent: no stored memory matched this query.",
            cited_memory_ids=[],
            memory_used=False,
            refused=True,
        )

    unsupported = [m for m in claimed_memory_ids if m not in fired]
    if unsupported:
        return GroundedResponse(
            text=(
                "Refused: the response cited memories that did not fire "
                f"({', '.join(unsupported)})."
            ),
            cited_memory_ids=[],
            memory_used=False,
            refused=True,
        )

    cited = [m for m in claimed_memory_ids if m in fired]
    if not cited:
        # SLM produced prose but grounded it in nothing; refuse.
        return GroundedResponse(
            text="Refused: response was not grounded in any fired memory.",
            cited_memory_ids=[],
            memory_used=False,
            refused=True,
        )

    return GroundedResponse(
        text=slm_text,
        cited_memory_ids=cited,
        memory_used=True,
        refused=False,
    )


def build_response(query_result: MemoryQueryResult,
                   bank: "MemoryBank") -> GroundedResponse:
    """Default grounding path used by the orchestrator for QUERY intents."""
    return ground_from_cells(query_result, bank)
