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
from agent.memory_ledger import MemoryLedger, SUPERSEDED
from agent.memory_lifecycle import (
    LifecycleVerdict,
    analyse_proposal_against_ledger,
    lexical_overlap,
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
from agent.pack_maintenance import (
    FreshnessPolicy,
    MaintenanceReport,
    RefreshItem,
    SourceFreshness,
    build_inventory,
    build_maintenance_report,
    build_refresh_plan,
)
from agent.project_packs import PackRegistry, ProjectPack
from agent.proposal_queue import ProposalQueue
from agent.query_planner import QueryRoute, plan_query
from agent.verifier import verify_candidate
from slm.schemas import VerificationVerdict, VerifiedMemoryCandidate
from slm.assistant_composer import (
    AssistantComposer,
    ComposedAnswer,
    ComposerMode,
    EvidenceItem,
    GroundingPackage,
    LocalSLMComposer,
    TemplateComposer,
)
from slm.answer_guard import enforce as guard_enforce
from slm.local_slm_backend import LocalSLMBackend, make_default_slm_backend

# Retrieval is intentionally permissive (epsilon large) so a stored memory is
# always offered as a candidate; safety is decided by the verifier, never by
# retrieval. These match scripts/demo_v1.py.
_EPSILON = 0.25
_RADIUS = 0.9
_K = 5

# Minimum content-token overlap for a superseded memory to surface in a
# historical query. Matches the duplicate threshold so history is shown only
# when the past memory is clearly about the same thing.
_HISTORICAL_OVERLAP = 0.4

# Staleness policies that surface an explicit "this source may be outdated"
# caution at query time. Sources marked ``review_required`` or ``static`` do not
# trigger this warning; only sources a curator has flagged as stale do, so the
# label distinguishes a known-stale source from a fresh one.
_STALE_POLICIES = {"stale", "outdated", "deprecated"}


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
    historical: List[dict] = field(default_factory=list)

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


@dataclass
class AssistantResult:
    """The optional assistant layer's full, auditable answer.

    It bundles the underlying combined audit (the trusted decision), the
    rendered :class:`ComposedAnswer`, the ids that were actually cited, the
    refusal flag, and the name of the composer backend that produced the prose.
    The composed answer can never contradict the audit: it is rendered from a
    :class:`GroundingPackage` derived from this same audit.
    """

    query: str
    audit: dict
    answer: ComposedAnswer
    evidence_ids: List[str]
    refused: bool
    composer_backend: str

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "audit": self.audit,
            "answer": self.answer.to_dict(),
            "evidence_ids": list(self.evidence_ids),
            "refused": self.refused,
            "composer_backend": self.composer_backend,
        }


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
                 semantic_embedder: Optional[EncoderProtocol] = None,
                 pack: Optional[ProjectPack] = None,
                 bank_path: Optional[str | Path] = None,
                 registry: Optional[PackRegistry] = None):
        self.encoder = encoder or DeterministicEncoder(dim=64)
        self.k = k
        # The v1.4 retrieval backend ranks imported knowledge. Default is the
        # deterministic, offline backend (tests, CI); ``semantic`` (env
        # KNOWLEDGE_RETRIEVAL_BACKEND or the constructor arg) uses MiniLM. If a
        # semantic backend is requested but unavailable, fall back to
        # deterministic so the workbench never breaks on a missing model.
        self._semantic_embedder = semantic_embedder
        # v1.6 project packs: when a pack is bound, every store is re-pointed at
        # the pack's own files and the bank is persisted so the pack's memories
        # survive across sessions. Without a pack, behaviour is exactly v1.5.
        self._registry = registry
        self._pack: Optional[ProjectPack] = None
        if pack is not None:
            self._bind_pack(pack, fresh=fresh, knowledge_backend=knowledge_backend)
        else:
            self._wire_stores(
                ledger_path=ledger_path, queue_path=queue_path,
                knowledge_path=knowledge_path, bank_path=bank_path,
                knowledge_backend=knowledge_backend, fresh=fresh)

    # -- store wiring / project packs (v1.6) --

    def _wire_stores(self, *, ledger_path, queue_path, knowledge_path,
                     bank_path, knowledge_backend, fresh: bool) -> None:
        """(Re)build the bank and the three stores at the given paths.

        This adds no geometry: the bank is the same frozen concept-cell
        substrate. When ``bank_path`` is set the bank is loaded from (and later
        persisted to) that file so a pack's minted ids stay aligned with its
        ledger across sessions. With no ``bank_path`` the bank is in-memory only,
        exactly as in v1.5.
        """
        self.bank = MemoryBank(self.encoder, epsilon=_EPSILON, radius=_RADIUS)
        self._bank_path = Path(bank_path) if bank_path else None
        if self._bank_path is not None and self._bank_path.exists() \
                and self._bank_path.stat().st_size > 0:
            self.bank.load(str(self._bank_path))
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
        self._knowledge_backend_requested = resolve_backend_name(knowledge_backend)
        self._knowledge_backend: RetrievalBackend = self._build_knowledge_backend()

    def _bind_pack(self, pack: ProjectPack, *, fresh: bool,
                   knowledge_backend: Optional[str]) -> None:
        """Point every store at ``pack``'s files and persist the bank there."""
        self._pack = pack
        backend = knowledge_backend or pack.default_knowledge_backend
        self._wire_stores(
            ledger_path=pack.memory_ledger_path,
            queue_path=pack.proposal_queue_path,
            knowledge_path=pack.knowledge_library_path,
            bank_path=pack.memory_bank_path,
            knowledge_backend=backend,
            fresh=fresh,
        )

    def _persist_bank(self) -> None:
        """Save the bank to the active pack's file (no-op without a pack)."""
        if self._bank_path is not None:
            self._bank_path.parent.mkdir(parents=True, exist_ok=True)
            self.bank.save(str(self._bank_path))

    @classmethod
    def from_pack(cls, pack: ProjectPack, *,
                  registry: Optional[PackRegistry] = None,
                  **kwargs) -> "WorkbenchService":
        """Build a service whose stores live entirely inside ``pack``."""
        return cls(pack=pack, registry=registry, **kwargs)

    def active_pack_info(self) -> Optional[dict]:
        """Return a summary of the bound pack, or ``None`` if global stores."""
        if self._pack is None:
            return None
        return self._pack.to_info_dict()

    def switch_pack(self, pack_id_or_name: str) -> ProjectPack:
        """Re-point every store at another pack and mark it active.

        Requires a :class:`PackRegistry` to have been supplied. The bank is
        rebuilt from the new pack's persisted records, so queries see only the
        new pack's memories.
        """
        if self._registry is None:
            raise ValueError("switch_pack requires a PackRegistry")
        pack = self._registry.get_pack(pack_id_or_name)
        if pack is None:
            raise KeyError(f"no such pack: {pack_id_or_name!r}")
        self._bind_pack(pack, fresh=False, knowledge_backend=None)
        self._registry.set_active_pack(pack.pack_id)
        return pack


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
        entry = self.ledger.add(rec.memory_id, rec.canonical_text,
                                source=source, tags=tags)
        self._persist_bank()
        return entry

    # -- query --

    def query_memory(self, query_text: str,
                     include_historical: bool = False) -> QueryAudit:
        """Run retrieve -> verify -> ground and return the audit trail.

        Grounding is always on the *current* memories only: a superseded memory
        was removed from the bank, so it can never be retrieved or cited as the
        answer. With ``include_historical=True`` the audit additionally carries a
        clearly-separated ``historical`` list of superseded memories whose text
        overlaps the query — surfaced as past record, never as a current answer.
        Deleted memories are never surfaced, even historically.
        """
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

        historical = (self._historical_matches(query_text)
                      if include_historical else [])

        return QueryAudit(
            query=query_text,
            candidate_retrieved=bool(candidates),
            verifier_verdict=_overall_verdict(views),
            memory_used=response.memory_used,
            refused=response.refused,
            response_text=response.text,
            cited_memory_ids=list(response.cited_memory_ids),
            candidates=views,
            historical=historical,
        )

    def _historical_matches(self, query_text: str) -> List[dict]:
        """Lexically scan superseded ledger entries relevant to the query.

        Superseded memories are gone from the bank (geometry deleted), so they
        cannot be retrieved as candidates; this is a plain text-overlap scan over
        the ledger's ``superseded`` entries only. Deleted entries are excluded so
        a deleted memory stays uncitable even in history.
        """
        matches: List[dict] = []
        for entry in self.ledger.entries():
            if entry.status != SUPERSEDED:
                continue
            overlap = lexical_overlap(query_text, entry.canonical_text)
            if overlap >= _HISTORICAL_OVERLAP:
                matches.append({
                    "memory_id": entry.memory_id,
                    "canonical_text": entry.canonical_text,
                    "status": entry.status,
                    "superseded_by": entry.superseded_by,
                    "overlap": round(overlap, 4),
                })
        matches.sort(key=lambda m: m["overlap"], reverse=True)
        return matches

    # -- delete --

    def delete_memory(self, memory_id: str) -> bool:
        """Remove a memory from the bank and mark the ledger entry deleted.

        Returns True if the memory existed and was removed. After deletion the
        cell is gone from the bank, so the memory can no longer be retrieved or
        cited; the ledger keeps a ``deleted`` record for audit.
        """
        removed = self.bank.delete(memory_id)
        marked = self.ledger.mark_deleted(memory_id)
        self._persist_bank()
        return removed or marked

    # -- dispute (v1.5 lifecycle, exposed for the review console) --

    def dispute_memory(self, memory_id: str,
                       conflict_id: Optional[str] = None) -> bool:
        """Flag a live memory as disputed, reusing the v1.5 ledger lifecycle.

        This adds no new lifecycle logic: it delegates to
        :meth:`MemoryLedger.mark_disputed`. The memory stays in the bank and
        remains citable; only the ledger status becomes ``disputed`` and, when
        ``conflict_id`` is given, the contradiction is cross-linked. Deleted or
        superseded memories cannot be disputed. Returns True if a current memory
        was flagged.
        """
        marked = self.ledger.mark_disputed(memory_id, conflict_id)
        if marked:
            self._persist_bank()
        return marked

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

    # -- v2.2 source maintenance (read-only) --------------------------------
    # These compute a freshness view over the active pack's sources. They
    # mutate nothing and change no grounding/citation/refusal semantics; they
    # only surface which sources need an operator's attention.

    def source_inventory(
            self, policy: "Optional[FreshnessPolicy]" = None
    ) -> "List[SourceFreshness]":
        """Freshness status for every active source in the active pack."""
        return build_inventory(self.knowledge, policy or FreshnessPolicy())

    def refresh_plan(
            self, policy: "Optional[FreshnessPolicy]" = None
    ) -> "List[RefreshItem]":
        """Action list for sources that are stale, review-due, or undatable."""
        return build_refresh_plan(self.source_inventory(policy))

    def maintenance_report(
            self, policy: "Optional[FreshnessPolicy]" = None, *,
            eval_total: Optional[int] = None,
            eval_passed: Optional[int] = None,
    ) -> "MaintenanceReport":
        """Source-health summary, optionally folding in eval pass numbers."""
        return build_maintenance_report(
            self.source_inventory(policy),
            eval_total=eval_total, eval_passed=eval_passed)

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
        cautions.extend(self._staleness_cautions(candidates))

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

    def _staleness_cautions(
            self, candidates: List[KnowledgeCandidate]) -> List[str]:
        """Warn when a cited source is flagged stale by its staleness policy."""
        cautions: List[str] = []
        warned: set[str] = set()
        for cand in candidates:
            if cand.source_name in warned:
                continue
            src = self.knowledge.get_source(cand.source_id)
            if src is None:
                continue
            policy = (src.staleness_policy or "").strip().lower()
            if policy in _STALE_POLICIES:
                warned.add(cand.source_name)
                cautions.append(
                    f"Source '{src.source_name}' is marked stale "
                    f"(staleness policy: {src.staleness_policy}); it may be "
                    "outdated \u2014 verify against current guidance before "
                    "relying on it.")
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

    # -- optional assistant composer layer (v2.0) --

    def build_grounding_package(
            self, query_text: str, *,
            allow_model_prior: bool = False) -> GroundingPackage:
        """Build the deterministic envelope the composer is allowed to render.

        Every decision here comes from the frozen path via :meth:`query_all`:
        which source grounded the answer, whether it was refused, and which ids
        may be cited. Only *grounded, accepted* memories and *retrieved*
        knowledge chunks become citable evidence. A near-miss the verifier
        rejected, and any superseded memory, are surfaced as display-only
        context that can never be cited.
        """
        combined = self.query_all(query_text)
        mem = combined.memory or {}
        know = combined.knowledge or {}

        evidence: List[EvidenceItem] = []
        # Memory evidence: strictly the grounded (accepted + cited) memories.
        mem_by_id = {c["memory_id"]: c for c in (mem.get("candidates") or [])}
        for mid in mem.get("cited_memory_ids") or []:
            cand = mem_by_id.get(mid)
            evidence.append(EvidenceItem(
                citation_id=f"mem:{mid}",
                kind="memory",
                text=(cand or {}).get("canonical_text", ""),
            ))
        # Knowledge evidence: the retrieved chunks (only present when the route
        # consulted knowledge and a chunk was returned).
        if combined.knowledge_used:
            for cand in know.get("candidates") or []:
                evidence.append(EvidenceItem(
                    citation_id=f"src:{cand['chunk_id']}",
                    kind="knowledge",
                    text=cand.get("text", ""),
                    source_name=cand.get("source_name"),
                    domain=cand.get("domain"),
                    authority=cand.get("authority"),
                ))

        informational = bool(know.get("informational_only"))
        conflict_context: List[EvidenceItem] = []
        historical_note: Optional[str] = None

        if combined.memory_used or combined.knowledge_used:
            mode = ComposerMode.GROUNDED
            refused = False
        elif mem.get("verifier_verdict") == "reject":
            mode = ComposerMode.CONFLICT_EXPLANATION
            refused = True
            for cand in mem.get("candidates") or []:
                conflict_context.append(EvidenceItem(
                    citation_id=f"rejected:{cand['memory_id']}",
                    kind="rejected",
                    text=cand.get("canonical_text", ""),
                ))
        elif allow_model_prior:
            mode = ComposerMode.MODEL_PRIOR_LABELLED
            refused = False
        else:
            mode = ComposerMode.REFUSAL
            refused = True

        # When not grounded, surface a superseded record (if any) as a
        # display-only note so the reviewer knows a past answer existed — but it
        # is never citable as the current answer.
        if mode in (ComposerMode.REFUSAL, ComposerMode.CONFLICT_EXPLANATION):
            hist = self.query_memory(
                query_text, include_historical=True).historical
            if hist:
                ids = ", ".join(h["memory_id"] for h in hist)
                historical_note = (
                    f"A superseded memory exists ({ids}) but is not current and "
                    "cannot be cited as the answer.")

        return GroundingPackage(
            query=query_text,
            mode=mode,
            memory_used=combined.memory_used,
            knowledge_used=combined.knowledge_used,
            model_prior_used=combined.model_prior_used,
            informational_only=informational,
            refused=refused,
            route=combined.route,
            evidence=evidence,
            conflict_context=conflict_context,
            cautions=list(combined.cautions),
            historical_note=historical_note,
            query_audit=combined.to_dict(),
        )

    def build_pack_summary_package(self) -> GroundingPackage:
        """Build a citable summary of the active pack's knowledge sources."""
        evidence: List[EvidenceItem] = []
        for src in self.list_knowledge_sources():
            evidence.append(EvidenceItem(
                citation_id=f"src:{src['source_id']}",
                kind="knowledge",
                text=src["source_name"],
                source_name=src["source_name"],
                domain=src["domain"],
                authority=src["authority"],
            ))
        return GroundingPackage(
            query="(pack summary)",
            mode=ComposerMode.PACK_SUMMARY,
            memory_used=False,
            knowledge_used=bool(evidence),
            model_prior_used=False,
            informational_only=False,
            refused=False,
            route="pack_summary",
            evidence=evidence,
        )

    def compose_answer(self, grounding_package: GroundingPackage,
                       composer: Optional[AssistantComposer] = None
                       ) -> ComposedAnswer:
        """Render a grounding package into prose with the given composer.

        The default composer is the deterministic template composer; nothing in
        the package can be overridden by the composer.
        """
        composer = composer or TemplateComposer()
        return composer.compose(grounding_package)

    def answer_query(self, query_text: str, use_slm: bool = False, *,
                     allow_model_prior: bool = False,
                     slm_backend: Optional[LocalSLMBackend] = None,
                     composer: Optional[AssistantComposer] = None
                     ) -> AssistantResult:
        """Answer a query with the optional composer layer.

        With ``use_slm=False`` (the default) the deterministic template composer
        renders the answer. With ``use_slm=True`` a :class:`LocalSLMComposer`
        wraps a local backend (the supplied one, or the environment default);
        if that backend is unavailable or misbehaves, the answer falls back to
        the template — so the assistant layer is always safe to call.
        """
        package = self.build_grounding_package(
            query_text, allow_model_prior=allow_model_prior)
        if composer is None:
            if use_slm:
                backend = slm_backend or make_default_slm_backend()
                composer = LocalSLMComposer(backend)
            else:
                composer = TemplateComposer()
        answer = self.compose_answer(package, composer)
        # Independent post-hoc guard: verify the composed answer never drifts
        # from the package (no invented citation, softened refusal, or
        # stale-as-current). On a REJECT the offending answer is discarded and
        # recomposed deterministically; the verdict is recorded in the audit.
        # The built-in template/SLM composers are correct by construction, so
        # this only changes output for a misbehaving custom composer.
        answer, guard_report = guard_enforce(package, answer)
        audit = dict(package.query_audit or {})
        audit["guard"] = guard_report.to_dict()
        return AssistantResult(
            query=query_text,
            audit=audit,
            answer=answer,
            evidence_ids=sorted(package.allowed_citation_ids),
            refused=answer.refused,
            composer_backend=answer.composer_backend,
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
        self._persist_bank()
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
