"""Workbench service: a thin orchestration layer over the frozen v1.0 path.

This is the single seam the v1.1 workbench (CLI or Streamlit) talks to. It owns
one ``MemoryBank`` (the frozen concept-cell substrate) and one ``MemoryLedger``
(human-facing provenance), and it composes the existing, untouched v1.0 pieces:

    retrieve_candidates  ->  verify_candidate  ->  ground_accepted_candidates

It adds no geometry, no new firing rule, and no new verifier logic. Its only job
is to turn a single user action into a structured, auditable result and to keep
the bank and ledger consistent (same minted ``memory_id`` in both, deletion
removes the cell *and* marks the ledger entry).

Every query returns a :class:`QueryAudit` whose fields are exactly the audit
trail the workbench surfaces: ``candidate_retrieved``, ``verifier_verdict``,
``memory_used``, and ``refused``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from agent.candidate_retrieval import (
    ground_accepted_candidates,
    retrieve_candidates,
)
from agent.memory_ledger import MemoryLedger
from agent.memory_lifecycle import (
    LifecycleVerdict,
    analyse_proposal_against_ledger,
)
from agent.knowledge_library import KnowledgeLibrary
from agent.knowledge_retrieval import KnowledgeCandidate, retrieve_knowledge
from agent.knowledge_sources import (
    KnowledgeDomain,
    KnowledgeSource,
    SourceAuthority,
    TRUSTED_AUTHORITIES,
)
from agent.retrieval_backends import (
    BackendUnavailableError,
    RetrievalBackend,
    make_backend,
    resolve_backend_name,
)
from agent.memory_proposals import ProposalBatch
from agent.note_ingestion import extract_candidate_memories, load_text_file
from agent.orchestrator import DeterministicEncoder, EncoderProtocol, MemoryBank
from agent.proposal_queue import ProposalQueue
from agent.query_planner import QueryRoute, plan_query
from agent.verifier import verify_candidate
from slm.schemas import VerificationVerdict, VerifiedMemoryCandidate

# Retrieval is intentionally permissive (epsilon large) so a stored memory is
# always offered as a candidate; safety is decided by the verifier, never by
# retrieval. These match scripts/demo_v1.py.
_EPSILON = 0.25
_RADIUS = 0.9
_K = 5


@dataclass
class CandidateView:
    """One retrieved candidate plus the verifier's verdict on it."""

    memory_id: str
    canonical_text: str
    activation: float
    rank: int
    verdict: str  # "accept" | "reject" | "ambiguous"


@dataclass
class QueryAudit:
    """The full, structured audit trail for a single query.

    The four load-bearing fields the workbench always shows are
    ``candidate_retrieved``, ``verifier_verdict``, ``memory_used`` and
    ``refused``. ``verifier_verdict`` is the *overall* verdict across candidates:
    ``accept`` if any candidate was accepted, else ``reject`` if any material
    mismatch was found, else ``ambiguous``, else ``none`` when nothing was
    retrieved.
    """

    query: str
    candidate_retrieved: bool
    verifier_verdict: str
    memory_used: bool
    refused: bool
    response_text: str
    cited_memory_ids: List[str] = field(default_factory=list)
    candidates: List[CandidateView] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _overall_verdict(views: List[CandidateView]) -> str:
    if not views:
        return "none"
    verdicts = {v.verdict for v in views}
    if "accept" in verdicts:
        return "accept"
    if "reject" in verdicts:
        return "reject"
    return "ambiguous"


@dataclass
class KnowledgeAudit:
    """The audit trail for a knowledge-only query.

    Imported knowledge is never project memory, so this result is kept apart
    from :class:`QueryAudit`: it reports which chunks were retrieved, the domain,
    whether the answer is restricted to informational use (medical/legal), and
    any provenance cautions (low authority, unknown coding version).
    """

    query: str
    knowledge_used: bool
    domain: Optional[str]
    informational_only: bool
    candidates: List[dict] = field(default_factory=list)
    cited_source_ids: List[str] = field(default_factory=list)
    cautions: List[str] = field(default_factory=list)
    response_text: str = ""
    backend_name: str = "deterministic"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CombinedAudit:
    """A query answered from memory and/or knowledge, kept clearly separated.

    ``memory_used`` is true only when a *project memory* was grounded;
    ``knowledge_used`` is true only when an *imported knowledge* chunk was
    retrieved; ``model_prior_used`` is true when neither grounded source could
    answer, so any reply would be the model's own ungrounded prior.
    """

    query: str
    route: str
    memory_used: bool
    knowledge_used: bool
    model_prior_used: bool
    memory: Optional[dict] = None
    knowledge: Optional[dict] = None
    cautions: List[str] = field(default_factory=list)
    knowledge_backend: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _domain_cautions(domain: Optional[str],
                     candidates: List[KnowledgeCandidate]) -> List[str]:
    """Build the query-time safety cautions for a knowledge answer.

    Medical/legal answers are always flagged informational only; medical answers
    add a no-diagnosis/no-prescription note. Low-authority sources (community /
    unknown) are warned about. Coding answers warn when the cited source has no
    recorded version.
    """
    cautions: List[str] = []
    if domain == KnowledgeDomain.MEDICAL.value:
        cautions.append(
            "Informational only: not medical advice, not a diagnosis, and not a "
            "prescription. Consult a qualified professional.")
    elif domain == KnowledgeDomain.LEGAL.value:
        cautions.append(
            "Informational only: not legal advice. Consult a qualified "
            "professional.")

    seen_low: set[str] = set()
    for cand in candidates:
        authority = SourceAuthority(cand.authority)
        if authority not in TRUSTED_AUTHORITIES and cand.source_name not in seen_low:
            seen_low.add(cand.source_name)
            cautions.append(
                f"Source '{cand.source_name}' has {authority.value} authority; "
                "verify before relying on it.")
    return cautions


class WorkbenchService:
    """Owns one memory bank + ledger and exposes the workbench operations."""

    def __init__(self, ledger_path: Optional[str | Path] = None,
                 encoder: Optional[EncoderProtocol] = None,
                 k: int = _K, fresh: bool = False,
                 queue_path: Optional[str | Path] = None,
                 knowledge_path: Optional[str | Path] = None,
                 knowledge_backend: Optional[str] = None,
                 semantic_embedder: Optional[EncoderProtocol] = None):
        self.encoder = encoder or DeterministicEncoder(dim=64)
        self.bank = MemoryBank(
            self.encoder,
            epsilon=_EPSILON,
            radius=_RADIUS,
        )
        # The v1.0 bank is in-memory only and mints its own ids, so it cannot be
        # restored from a saved ledger. ``fresh`` starts the ledger empty (and
        # overwrites any prior file on first write) to keep bank and ledger ids
        # aligned; the workbench app uses it because each launch rebuilds the
        # bank from scratch.
        self.ledger = MemoryLedger(ledger_path, load_existing=not fresh)
        # The proposal queue is the v1.2 ingestion holding area: candidate
        # memories live here until a human approves them. It follows the same
        # ``fresh`` rule as the ledger so a relaunch does not double-apply a
        # queue whose approved memories were already written.
        self.proposals = ProposalQueue(queue_path, load_existing=not fresh)
        # The knowledge library is the v1.3 import store: external sources and
        # their chunks live here, fully separate from the memory bank/ledger so
        # imported knowledge can never be mistaken for a project decision. It
        # follows the same ``fresh`` rule as the ledger and queue.
        self.knowledge = KnowledgeLibrary(knowledge_path, load_existing=not fresh)
        self.k = k
        # The v1.4 retrieval backend ranks imported knowledge. Default is the
        # deterministic, offline backend (tests, CI); ``semantic`` (env
        # KNOWLEDGE_RETRIEVAL_BACKEND or the constructor arg) uses MiniLM. If a
        # semantic backend is requested but unavailable, fall back to
        # deterministic so the workbench never breaks on a missing model.
        self._semantic_embedder = semantic_embedder
        self._knowledge_backend_requested = resolve_backend_name(knowledge_backend)
        self._knowledge_backend: RetrievalBackend = self._build_knowledge_backend()

    def _build_knowledge_backend(self) -> RetrievalBackend:
        """Build the requested knowledge backend, falling back gracefully."""
        try:
            return make_backend(
                self._knowledge_backend_requested,
                deterministic_encoder=self.encoder,
                semantic_embedder=self._semantic_embedder,
            )
        except BackendUnavailableError:
            return make_backend(
                "deterministic", deterministic_encoder=self.encoder)

    def knowledge_backend_name(self) -> str:
        """Return the name of the **active** knowledge retrieval backend.

        This is the backend actually in use, which may differ from the one
        requested if a semantic backend was asked for but was unavailable and
        the service fell back to deterministic.
        """
        return self._knowledge_backend.backend_name

    # -- write --

    def add_memory(self, text: str, source: Optional[str] = None,
                   tags: Optional[List[str]] = None):
        """Write a memory into the bank and record it in the ledger.

        Returns the created :class:`~agent.memory_ledger.LedgerEntry`. The bank
        mints the ``memory_id``; the ledger records the same id so the two stay
        aligned.
        """
        rec = self.bank.write(text, source=source or "user", tags=tags or [])
        return self.ledger.add(rec.memory_id, rec.canonical_text,
                               source=source, tags=tags)

    # -- query --

    def query_memory(self, query_text: str) -> QueryAudit:
        """Run retrieve -> verify -> ground and return the audit trail."""
        candidates = retrieve_candidates(self.bank, query_text, self.k)

        verified: List[VerifiedMemoryCandidate] = []
        views: List[CandidateView] = []
        for cand in candidates:
            verdict = verify_candidate(query_text, cand.canonical_text)
            verified.append(
                VerifiedMemoryCandidate(candidate=cand, verdict=verdict)
            )
            views.append(CandidateView(
                memory_id=cand.memory_id,
                canonical_text=cand.canonical_text,
                activation=cand.activation,
                rank=cand.rank,
                verdict=verdict.value,
            ))

        response = ground_accepted_candidates(verified)

        return QueryAudit(
            query=query_text,
            candidate_retrieved=bool(candidates),
            verifier_verdict=_overall_verdict(views),
            memory_used=response.memory_used,
            refused=response.refused,
            response_text=response.text,
            cited_memory_ids=list(response.cited_memory_ids),
            candidates=views,
        )

    # -- delete --

    def delete_memory(self, memory_id: str) -> bool:
        """Remove a memory from the bank and mark the ledger entry deleted.

        Returns True if the memory existed and was removed. After deletion the
        cell is gone from the bank, so the memory can no longer be retrieved or
        cited; the ledger keeps a ``deleted`` record for audit.
        """
        removed = self.bank.delete(memory_id)
        marked = self.ledger.mark_deleted(memory_id)
        return removed or marked

    # -- knowledge import / query (v1.3) --

    def import_knowledge(self, path: str | Path, *,
                         domain: str | KnowledgeDomain,
                         authority: str | SourceAuthority,
                         source_name: str,
                         version: Optional[str] = None) -> KnowledgeSource:
        """Import an external document into the knowledge library.

        The document is chunked and stored with its provenance; **nothing** is
        written to the memory bank or ledger. Returns the created
        :class:`KnowledgeSource`.
        """
        dom = domain if isinstance(domain, KnowledgeDomain) else KnowledgeDomain(domain)
        auth = (authority if isinstance(authority, SourceAuthority)
                else SourceAuthority(authority))
        return self.knowledge.import_text_file(
            path, domain=dom, authority=auth,
            source_name=source_name, version=version,
        )

    def list_knowledge_sources(self) -> List[dict]:
        """Return active knowledge sources as plain dicts for display/export."""
        rows: List[dict] = []
        for src in self.knowledge.list_sources(active_only=True):
            rows.append({
                "source_id": src.source_id,
                "source_name": src.source_name,
                "domain": src.domain.value,
                "authority": src.authority.value,
                "version": src.version,
                "path_or_url": src.path_or_url,
                "chunks": len(self.knowledge.list_chunks(source_id=src.source_id)),
            })
        return rows

    def delete_knowledge_source(self, source_id: str) -> bool:
        """Deactivate a knowledge source so its chunks can no longer be cited."""
        return self.knowledge.delete_source(source_id)

    def query_knowledge(self, query_text: str) -> KnowledgeAudit:
        """Retrieve imported knowledge chunks for ``query_text``.

        Only the knowledge library is consulted; no project memory is touched,
        so ``memory_used`` is irrelevant here. Domain safety rules are applied:
        medical/legal answers are flagged informational only, low-authority and
        unknown-version sources are cautioned.
        """
        plan = plan_query(query_text)
        active_chunks = self.knowledge.list_chunks(active_only=True)
        source_versions = {
            src.source_id: src.version
            for src in self.knowledge.list_sources(active_only=True)
            if src.version
        }
        candidates = retrieve_knowledge(
            self.encoder, active_chunks, query_text, self.k,
            backend=self._knowledge_backend, source_versions=source_versions)

        domain = plan.domain.value if plan.domain else (
            candidates[0].domain if candidates else None)
        informational = domain in {
            KnowledgeDomain.MEDICAL.value, KnowledgeDomain.LEGAL.value}

        cautions = _domain_cautions(domain, candidates)
        cautions.extend(self._coding_version_cautions(candidates))

        cited: List[str] = []
        cand_dicts: List[dict] = []
        for cand in candidates:
            if cand.source_id not in cited:
                cited.append(cand.source_id)
            cand_dicts.append({
                "source_id": cand.source_id,
                "chunk_id": cand.chunk_id,
                "source_name": cand.source_name,
                "domain": cand.domain,
                "authority": cand.authority,
                "version": cand.version,
                "activation": cand.activation,
                "rank": cand.rank,
                "section": cand.source_section,
                "text": cand.chunk_text,
                "backend_name": cand.backend_name,
                "is_semantic": cand.is_semantic,
            })

        if candidates:
            lines = [f"- ({c.source_name}) {c.chunk_text}" for c in candidates]
            response = "From imported knowledge:\n" + "\n".join(lines)
        else:
            response = "Knowledge is silent: no imported source matched."

        return KnowledgeAudit(
            query=query_text,
            knowledge_used=bool(candidates),
            domain=domain,
            informational_only=informational,
            candidates=cand_dicts,
            cited_source_ids=cited,
            cautions=cautions,
            response_text=response,
            backend_name=self.knowledge_backend_name(),
        )

    def _coding_version_cautions(
            self, candidates: List[KnowledgeCandidate]) -> List[str]:
        """Warn when a cited coding source has no recorded version."""
        cautions: List[str] = []
        warned: set[str] = set()
        for cand in candidates:
            if cand.domain != KnowledgeDomain.CODING.value:
                continue
            src = self.knowledge.get_source(cand.source_id)
            if src is None or cand.source_name in warned:
                continue
            warned.add(cand.source_name)
            if src.version:
                cautions.append(
                    f"Coding source '{src.source_name}' is version {src.version}.")
            else:
                cautions.append(
                    f"Coding source '{src.source_name}' has no recorded "
                    "version; behaviour may differ across versions.")
        return cautions

    def query_all(self, query_text: str) -> CombinedAudit:
        """Answer a query from memory and/or knowledge, kept clearly separated.

        The planner decides where to look. Memory and knowledge are queried
        through their own independent paths, so a knowledge chunk can never be
        cited as a project decision and a memory can never be returned as an
        external reference. ``model_prior_used`` is set when neither grounded
        source could answer.
        """
        plan = plan_query(query_text)

        memory_audit: Optional[QueryAudit] = None
        knowledge_audit: Optional[KnowledgeAudit] = None

        want_memory = plan.route in {
            QueryRoute.MEMORY_ONLY, QueryRoute.BOTH}
        want_knowledge = plan.route in {
            QueryRoute.KNOWLEDGE_ONLY, QueryRoute.BOTH}

        if want_memory:
            memory_audit = self.query_memory(query_text)
        if want_knowledge:
            knowledge_audit = self.query_knowledge(query_text)

        memory_used = bool(memory_audit and memory_audit.memory_used)
        knowledge_used = bool(knowledge_audit and knowledge_audit.knowledge_used)
        model_prior_used = not memory_used and not knowledge_used

        cautions = list(knowledge_audit.cautions) if knowledge_audit else []
        if model_prior_used:
            cautions.append(
                "Neither project memory nor imported knowledge answered; any "
                "reply would be the model's ungrounded prior.")

        route = (QueryRoute.GENERAL_MODEL_NOT_GROUNDED.value
                 if model_prior_used else plan.route.value)

        return CombinedAudit(
            query=query_text,
            route=route,
            memory_used=memory_used,
            knowledge_used=knowledge_used,
            model_prior_used=model_prior_used,
            memory=memory_audit.to_dict() if memory_audit else None,
            knowledge=knowledge_audit.to_dict() if knowledge_audit else None,
            cautions=cautions,
            knowledge_backend=self.knowledge_backend_name(),
        )

    # -- ingestion / approval queue (v1.2) --

    def import_notes(self, path: str | Path) -> ProposalBatch:
        """Import a note, extract candidate memories, and queue them.

        Extraction is deterministic and offline (no LLM). The returned batch
        contains every candidate; the proposals are queued as ``pending`` and
        nothing is written to the bank until a human approves them.
        """
        source_file = str(path)
        text = load_text_file(path)
        batch = extract_candidate_memories(text, source_file)
        self.proposals.add_batch(batch)
        # Tag each new proposal with how it relates to the existing memories
        # (new / duplicate / conflict) so a reviewer sees it before approving.
        for proposal in batch.proposals:
            self._record_lifecycle(proposal)
        return batch

    def list_proposals(self, status: Optional[str] = "pending"):
        """List queued proposals, filtered by status (``None`` for all)."""
        return self.proposals.list(status)

    # -- lifecycle analysis (v1.3) --

    def _candidate_retriever(self, text: str):
        return retrieve_candidates(self.bank, text, self.k)

    def _record_lifecycle(self, proposal):
        """Run lifecycle analysis for one proposal and store the result."""
        check = analyse_proposal_against_ledger(
            proposal, self.ledger,
            candidate_retriever=self._candidate_retriever,
        )
        self.proposals.set_lifecycle(
            proposal.proposal_id,
            check.verdict.value,
            check.candidate_memory_id,
            check.reason,
        )
        return check

    def analyse_proposal(self, proposal_id: str):
        """Re-run lifecycle analysis for a queued proposal and store it.

        Returns the :class:`LifecycleCheck`, or ``None`` if the id is unknown.
        """
        proposal = self.proposals.get(proposal_id)
        if proposal is None:
            return None
        return self._record_lifecycle(proposal)

    def list_conflicts(self):
        """Pending proposals flagged as a conflict (hard or possible)."""
        flagged = {LifecycleVerdict.CONFLICT.value,
                   LifecycleVerdict.POSSIBLE_CONFLICT.value}
        return [p for p in self.proposals.list_pending()
                if p.lifecycle_verdict in flagged]

    def list_duplicates(self):
        """Pending proposals flagged as a duplicate (exact or possible)."""
        flagged = {LifecycleVerdict.DUPLICATE.value,
                   LifecycleVerdict.POSSIBLE_DUPLICATE.value}
        return [p for p in self.proposals.list_pending()
                if p.lifecycle_verdict in flagged]

    def approve_proposal(self, proposal_id: str) -> bool:
        """Approve a proposal so it will be written by ``write_approved``.

        A proposal flagged as a hard ``duplicate`` or ``conflict`` is refused
        here: writing it needs the explicit :meth:`approve_proposal_as_new` or
        :meth:`approve_proposal_superseding`. Soft (possible) flags are allowed.
        """
        proposal = self.proposals.get(proposal_id)
        if proposal is not None and proposal.lifecycle_verdict in (
            LifecycleVerdict.DUPLICATE.value,
            LifecycleVerdict.CONFLICT.value,
        ):
            return False
        return self.proposals.approve(proposal_id)

    def approve_proposal_as_new(self, proposal_id: str) -> bool:
        """Explicitly approve a proposal as a new memory, despite any flag.

        This is the user-forced path for a duplicate or conflict: the proposal
        is written as its own new memory and supersedes nothing.
        """
        self.proposals.set_supersedes(proposal_id, None)
        return self.proposals.approve(proposal_id)

    def approve_proposal_superseding(self, proposal_id: str,
                                     old_memory_id: str) -> bool:
        """Approve a proposal that will supersede ``old_memory_id`` when written.

        The old memory must exist and be active. On write the old memory is
        marked ``superseded`` and removed from the bank so it can no longer be
        retrieved or grounded as the current answer.
        """
        if self.ledger.get(old_memory_id) is None:
            return False
        self.proposals.set_supersedes(proposal_id, old_memory_id)
        return self.proposals.approve(proposal_id)

    def reject_proposal(self, proposal_id: str) -> bool:
        """Reject a proposal so it is never written to the bank."""
        return self.proposals.reject(proposal_id)

    def edit_proposal(self, proposal_id: str, new_text: str) -> bool:
        """Edit a proposal's text (status becomes ``edited``, audit preserved)."""
        return self.proposals.edit(proposal_id, new_text)

    def write_approved_proposals(self) -> List:
        """Write every approved, not-yet-written proposal into the bank/ledger.

        Approved proposals reuse the frozen :meth:`add_memory` path, so the bank
        and ledger stay aligned and the verifier/grounding policy is unchanged.
        Rejected proposals are skipped entirely. Each written proposal is marked
        so a second call cannot create duplicate memories. Returns the created
        ledger entries.
        """
        written = []
        for proposal in self.proposals.approved_unwritten():
            tags = list(proposal.tags)
            if proposal.kind.value not in tags:
                tags.append(proposal.kind.value)
            entry = self.add_memory(
                proposal.canonical_text,
                source=proposal.source_file,
                tags=tags,
            )
            # If this proposal supersedes an older memory, mark the old one
            # superseded and remove it from the bank so it can no longer be
            # retrieved or grounded as the current answer (the ledger keeps the
            # superseded record and the supersession chain).
            if proposal.supersedes_memory_id:
                self.ledger.mark_superseded(
                    proposal.supersedes_memory_id, entry.memory_id)
                self.bank.delete(proposal.supersedes_memory_id)
            self.proposals.mark_written(proposal.proposal_id)
            written.append(entry)
        return written

    # -- inspect / export --

    def export_ledger(self) -> List[dict]:
        """Return the full ledger (active and deleted) as plain dicts."""
        return self.ledger.export()

    def export_proposals(self) -> List[dict]:
        """Return the whole proposal queue as plain dicts."""
        return self.proposals.export()

    def seed_from(self, seed_path: str | Path) -> List[str]:
        """Load seed memories from a JSONL file, writing each into the bank.

        Each line is a JSON object with at least ``canonical_text`` and optional
        ``source`` and ``tags`` (any ``memory_id`` in the file is ignored so the
        bank and ledger share freshly minted, aligned ids). Returns the list of
        minted memory ids.
        """
        path = Path(seed_path)
        minted: List[str] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                text = data["canonical_text"]
                entry = self.add_memory(
                    text,
                    source=data.get("source"),
                    tags=list(data.get("tags", [])),
                )
                minted.append(entry.memory_id)
        return minted
