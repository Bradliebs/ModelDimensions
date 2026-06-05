"""Deterministic, read-only active-pack monitoring and regression detection (v7.1).

A knowledge pack becomes *active* only through the governed activation lifecycle
(see :mod:`agent.knowledge_pack_activation`). Activation binds an approval to an
exact pack fingerprint **and** an exact evaluation fingerprint, and a passing
evaluation is evidence — never approval. This module adds the *next* governed
step: once a pack is active and serving retrieval, monitoring continuously
re-measures the active set against a pre-activation (or other fixed) baseline and
**detects regressions** in retrieval, evidence selection, citation integrity,
source isolation and unrelated-query behaviour.

Cardinal rules (enforced + tested):

* Monitoring is **read-only**. It never activates, deactivates, supersedes or
  rolls back a pack, and never mutates active-pack state, pack contents, retrieval
  indexes, the source registry, a MemoryLedger or any proposal.
* Monitoring evidence is **not** approval. A rollback *recommendation* is not a
  rollback *approval*. Recommendations are advisory only — a human acts on them
  through the separately governed lifecycle.
* A passing aggregate score must never hide a critical-case failure. Critical
  cases are evaluated separately from aggregate averages.
* Every monitoring result binds to the exact active-pack snapshot, the exact
  monitoring policy, the exact eval-corpus fingerprint, the retrieval-configuration
  fingerprint and the evaluator version. A stored result is *invalid* once any of
  those change.
* Attribution is conservative: a regression is attributed to a specific pack only
  when disabling that pack (in an isolated evaluation, never the live state)
  removes the regression while configuration and corpus are unchanged and repeated
  runs agree. Otherwise the cause is classified conservatively.

Boundaries (verified by an import-purity test): this module imports **only the
standard library** plus a small set of *read-only* names from
:mod:`agent.knowledge_pack_activation` (state/identity models, the lifecycle enum
and the active-set selection adapter). It imports no activation/deactivation/
rollback writer, no MemoryLedger writer, no source-registry writer, no proposal
applier, no pack-content or retrieval-index writer, and no LLM client. The only
thing it writes is an append-only monitoring-history file, and then only when a
caller passes an explicit path/flag.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from agent.knowledge_pack_activation import (
    ACTIVATION_LAYER_VERSION,
    ActivationStateManifest,
    ActivePackState,
    CoexistencePolicy,
    PackLifecycleState,
    select_active_pack_ids,
)

MONITOR_LAYER_VERSION = "active-pack-monitor-v7.1"
EVALUATOR_VERSION = "monitor-eval-v7.1"

SNAPSHOT_FP_PREFIX = "activeset-"
CORPUS_FP_PREFIX = "moncorpus-"
RETRIEVAL_CFG_PREFIX = "retrcfg-"
BASELINE_HASH_PREFIX = "monbase-"
RUN_HASH_PREFIX = "monrun-"
RECORD_HASH_PREFIX = "monrec-"

DEFAULT_HISTORY_PATH = "reports/active_pack_monitoring.jsonl"

# Default retrieval depth used by the wrong-source / hit metrics.
DEFAULT_HIT_K = 5


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mean(values: Sequence[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _round(value: Optional[float], places: int = 4) -> Optional[float]:
    return None if value is None else round(value, places)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MonitoringSeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


_SEVERITY_RANK = {
    MonitoringSeverity.CRITICAL: 0,
    MonitoringSeverity.HIGH: 1,
    MonitoringSeverity.MEDIUM: 2,
    MonitoringSeverity.LOW: 3,
    MonitoringSeverity.INFO: 4,
}


class MonitoringCaseClass(str, Enum):
    CRITICAL = "critical_cases"
    PACK_RELEVANT = "pack_relevant_cases"
    UNRELATED_CONTROL = "unrelated_control_cases"
    FORBIDDEN_SOURCE = "forbidden_source_cases"
    CITATION_INTEGRITY = "citation_integrity_cases"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence_cases"
    SUPERSESSION = "supersession_cases"
    NEAR_NEIGHBOUR = "near_neighbour_cases"


class BaselineType(str, Enum):
    PRE_ACTIVATION = "pre_activation"
    PREVIOUS_ACTIVE_STATE = "previous_active_state"
    KNOWN_GOOD_STATE = "known_good_state"
    ROLLING_REFERENCE = "rolling_reference"
    FIXED_RELEASE_BASELINE = "fixed_release_baseline"


class RegressionFindingCode(str, Enum):
    EXPECTED_SOURCE_DROPPED = "expected_source_dropped"
    EXPECTED_CHUNK_DROPPED = "expected_chunk_dropped"
    EXPECTED_SOURCE_RANK_WORSENED = "expected_source_rank_worsened"
    WRONG_SOURCE_RATE_INCREASED = "wrong_source_rate_increased"
    FORBIDDEN_SOURCE_INTRODUCED = "forbidden_source_introduced"
    UNRELATED_CASE_REGRESSED = "unrelated_case_regressed"
    INACTIVE_PACK_RETRIEVED = "inactive_pack_retrieved"
    SUPERSEDED_PACK_RETRIEVED = "superseded_pack_retrieved"
    RETIRED_PACK_RETRIEVED = "retired_pack_retrieved"
    ENVIRONMENT_SCOPE_VIOLATION = "environment_scope_violation"
    UNSUPPORTED_CITATION_INTRODUCED = "unsupported_citation_introduced"
    UNCITED_CLAIM_INTRODUCED = "uncited_claim_introduced"
    CITATION_LINEAGE_DEGRADED = "citation_lineage_degraded"
    FALSE_ANSWERABLE_INTRODUCED = "false_answerable_introduced"
    FALSE_INSUFFICIENT_INTRODUCED = "false_insufficient_introduced"
    DUPLICATE_INTERFERENCE_INCREASED = "duplicate_interference_increased"
    INSTABILITY_DETECTED = "instability_detected"
    CRITICAL_CASE_FAILED = "critical_case_failed"
    AGGREGATE_IMPROVED_BUT_CRITICAL_FAILED = "aggregate_improved_but_critical_failed"
    INSUFFICIENT_SAMPLE = "insufficient_sample"
    BASELINE_INCOMPATIBLE = "baseline_incompatible"
    MONITORING_RUN_INVALID = "monitoring_run_invalid"


# Finding codes whose mere presence is a critical safety signal.
_CRITICAL_FINDING_CODES = frozenset({
    RegressionFindingCode.FORBIDDEN_SOURCE_INTRODUCED,
    RegressionFindingCode.INACTIVE_PACK_RETRIEVED,
    RegressionFindingCode.SUPERSEDED_PACK_RETRIEVED,
    RegressionFindingCode.RETIRED_PACK_RETRIEVED,
    RegressionFindingCode.ENVIRONMENT_SCOPE_VIOLATION,
    RegressionFindingCode.CRITICAL_CASE_FAILED,
    RegressionFindingCode.AGGREGATE_IMPROVED_BUT_CRITICAL_FAILED,
})


class AttributionClass(str, Enum):
    PACK_SPECIFIC = "pack_specific"
    ACTIVE_SET_INTERACTION = "active_set_interaction"
    RETRIEVAL_CONFIGURATION = "retrieval_configuration"
    CORPUS_CHANGE = "corpus_change"
    EVALUATOR_CHANGE = "evaluator_change"
    UNSTABLE_RESULT = "unstable_result"
    UNDETERMINED = "undetermined"


class MonitoringRecommendationCode(str, Enum):
    KEEP_ACTIVE = "keep_active"
    KEEP_ACTIVE_WITH_WATCH = "keep_active_with_watch"
    INVESTIGATE = "investigate"
    DEACTIVATE_RECOMMENDED = "deactivate_recommended"
    ROLLBACK_RECOMMENDED = "rollback_recommended"
    BLOCK_FUTURE_ACTIVATION = "block_future_activation"
    INSUFFICIENT_EVIDENCE_TO_RECOMMEND = "insufficient_evidence_to_recommend"


class ConfidenceBand(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ---------------------------------------------------------------------------
# Monitoring case corpus
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonitoringCase:
    """One monitoring probe. Reuses the retrieval-eval case vocabulary."""

    case_id: str
    query: str
    case_class: MonitoringCaseClass = MonitoringCaseClass.PACK_RELEVANT
    severity: MonitoringSeverity = MonitoringSeverity.MEDIUM
    expected_sources: Tuple[str, ...] = ()
    expected_chunks: Tuple[str, ...] = ()
    forbidden_sources: Tuple[str, ...] = ()
    forbidden_chunks: Tuple[str, ...] = ()
    expected_answerability: Optional[bool] = None
    expected_citation_behaviour: str = ""
    relevant_pack_ids: Tuple[str, ...] = ()
    control_case: bool = False
    critical_case: bool = False
    minimum_hit_k: int = DEFAULT_HIT_K
    notes: str = ""

    @property
    def is_critical(self) -> bool:
        return self.critical_case or self.severity == MonitoringSeverity.CRITICAL

    def canonical(self) -> dict:
        return {
            "case_id": self.case_id,
            "query": self.query,
            "case_class": self.case_class.value,
            "severity": self.severity.value,
            "expected_sources": list(self.expected_sources),
            "expected_chunks": list(self.expected_chunks),
            "forbidden_sources": list(self.forbidden_sources),
            "forbidden_chunks": list(self.forbidden_chunks),
            "expected_answerability": self.expected_answerability,
            "expected_citation_behaviour": self.expected_citation_behaviour,
            "relevant_pack_ids": list(self.relevant_pack_ids),
            "control_case": self.control_case,
            "critical_case": self.critical_case,
            "minimum_hit_k": self.minimum_hit_k,
        }

    def to_dict(self) -> dict:
        return {"_record": "active_pack_monitor_case", **self.canonical(),
                "notes": self.notes}

    @classmethod
    def from_dict(cls, data: dict) -> "MonitoringCase":
        def _tuple(key: str) -> Tuple[str, ...]:
            value = data.get(key) or []
            return tuple(str(v) for v in value)

        answerability = data.get("expected_answerability")
        return cls(
            case_id=str(data.get("case_id") or data.get("query", ""))[:80],
            query=str(data.get("query", "")),
            case_class=MonitoringCaseClass(
                str(data.get("case_class", "pack_relevant_cases"))),
            severity=MonitoringSeverity(str(data.get("severity", "medium"))),
            expected_sources=_tuple("expected_sources"),
            expected_chunks=_tuple("expected_chunks"),
            forbidden_sources=_tuple("forbidden_sources"),
            forbidden_chunks=_tuple("forbidden_chunks"),
            expected_answerability=(
                None if answerability is None else bool(answerability)),
            expected_citation_behaviour=str(
                data.get("expected_citation_behaviour", "")),
            relevant_pack_ids=_tuple("relevant_pack_ids"),
            control_case=bool(data.get("control_case", False)),
            critical_case=bool(data.get("critical_case", False)),
            minimum_hit_k=int(data.get("minimum_hit_k", DEFAULT_HIT_K)),
            notes=str(data.get("notes", "")),
        )


def load_monitoring_cases(path: str | Path) -> List[MonitoringCase]:
    """Load monitoring cases from a JSONL file (``#`` comment lines allowed)."""
    cases: List[MonitoringCase] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        cases.append(MonitoringCase.from_dict(json.loads(line)))
    return cases


def compute_corpus_fingerprint(cases: Sequence[MonitoringCase]) -> str:
    """Deterministic fingerprint over the monitoring corpus (order-independent)."""
    payload = sorted(_canonical(c.canonical()) for c in cases)
    return CORPUS_FP_PREFIX + _sha256_hex(_canonical(payload))[:16]


def compute_retrieval_config_fingerprint(*, backend: str, hit_k: int = DEFAULT_HIT_K,
                                         extra: Optional[dict] = None) -> str:
    """Fingerprint of the retrieval configuration a run/baseline was measured under."""
    payload = {
        "activation_layer": ACTIVATION_LAYER_VERSION,
        "backend": backend,
        "hit_k": hit_k,
        "evaluator": EVALUATOR_VERSION,
        "extra": extra or {},
    }
    return RETRIEVAL_CFG_PREFIX + _sha256_hex(_canonical(payload))[:12]


# ---------------------------------------------------------------------------
# Stage observations + per-case result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonitoringStageObservation:
    """What one monitored stage (raw / selected / cited) actually surfaced.

    Stores only IDs, ranks and counts — never full retrieved text.
    """

    stage: str  # "raw" | "selected" | "cited"
    source_names: Tuple[str, ...] = ()
    chunk_ids: Tuple[str, ...] = ()
    pack_ids: Tuple[str, ...] = ()
    source_revisions: Tuple[str, ...] = ()
    first_expected_rank: Optional[int] = None
    forbidden_source_hits: Tuple[str, ...] = ()
    answerable: bool = True

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "source_names": list(self.source_names),
            "chunk_ids": list(self.chunk_ids),
            "pack_ids": list(self.pack_ids),
            "source_revisions": list(self.source_revisions),
            "first_expected_rank": self.first_expected_rank,
            "forbidden_source_hits": list(self.forbidden_source_hits),
            "answerable": self.answerable,
        }


@dataclass(frozen=True)
class MonitoringCaseResult:
    """Measured, read-only facts for one case under one active-pack snapshot.

    All safety facts are stored as explicit counts/booleans so the comparison,
    attribution and recommendation logic is pure and deterministic. The optional
    stage observations carry the lineage IDs that produced those facts.
    """

    case_id: str
    case_class: MonitoringCaseClass
    severity: MonitoringSeverity
    critical_case: bool = False
    control_case: bool = False

    # Retrieval (raw stage).
    hit: Optional[bool] = None
    first_expected_rank: Optional[int] = None
    expected_source_recall: Optional[float] = None
    expected_chunk_recall: Optional[float] = None
    wrong_source_rate: Optional[float] = None
    raw_forbidden_source_hit_count: int = 0
    off_topic_inclusion: bool = False
    duplicate_interference_rate: float = 0.0

    # Evidence selection (selected stage; relevance-gated).
    selected_expected_source_recall: Optional[float] = None
    selected_wrong_source_rate: Optional[float] = None
    selected_forbidden_source_count: int = 0

    # Citations (cited stage).
    cited_expected_source_recall: Optional[float] = None
    cited_forbidden_source_count: int = 0
    unsupported_citation_count: int = 0
    uncited_factual_claim_count: int = 0
    citation_lineage_accuracy: Optional[float] = None

    # Source isolation (against the active-pack snapshot).
    inactive_pack_hit_count: int = 0
    superseded_pack_hit_count: int = 0
    retired_pack_hit_count: int = 0
    environment_scope_violation_count: int = 0

    # Sufficiency.
    false_answerable: bool = False
    false_insufficient: bool = False
    correct_insufficient: bool = False

    # Stability (repeated runs).
    repeated_run_disagreement_count: int = 0

    passed: bool = True
    pack_ids_observed: Tuple[str, ...] = ()
    raw: Optional[MonitoringStageObservation] = None
    selected: Optional[MonitoringStageObservation] = None
    cited: Optional[MonitoringStageObservation] = None

    @property
    def is_critical(self) -> bool:
        return self.critical_case or self.severity == MonitoringSeverity.CRITICAL

    @property
    def stable(self) -> bool:
        return self.repeated_run_disagreement_count == 0

    @property
    def gated_forbidden_hit_count(self) -> int:
        """Forbidden presence at the stages the relevance gate can enforce."""
        return self.selected_forbidden_source_count + self.cited_forbidden_source_count

    @property
    def inactive_or_superseded_hit_count(self) -> int:
        return (self.inactive_pack_hit_count + self.superseded_pack_hit_count
                + self.retired_pack_hit_count)

    def to_dict(self) -> dict:
        data = {
            "_record": "active_pack_monitor_case_result",
            "case_id": self.case_id,
            "case_class": self.case_class.value,
            "severity": self.severity.value,
            "critical_case": self.is_critical,
            "control_case": self.control_case,
            "hit": self.hit,
            "first_expected_rank": self.first_expected_rank,
            "expected_source_recall": _round(self.expected_source_recall),
            "expected_chunk_recall": _round(self.expected_chunk_recall),
            "wrong_source_rate": _round(self.wrong_source_rate),
            "raw_forbidden_source_hit_count": self.raw_forbidden_source_hit_count,
            "off_topic_inclusion": self.off_topic_inclusion,
            "duplicate_interference_rate": _round(self.duplicate_interference_rate),
            "selected_expected_source_recall": _round(self.selected_expected_source_recall),
            "selected_wrong_source_rate": _round(self.selected_wrong_source_rate),
            "selected_forbidden_source_count": self.selected_forbidden_source_count,
            "cited_expected_source_recall": _round(self.cited_expected_source_recall),
            "cited_forbidden_source_count": self.cited_forbidden_source_count,
            "unsupported_citation_count": self.unsupported_citation_count,
            "uncited_factual_claim_count": self.uncited_factual_claim_count,
            "citation_lineage_accuracy": _round(self.citation_lineage_accuracy),
            "inactive_pack_hit_count": self.inactive_pack_hit_count,
            "superseded_pack_hit_count": self.superseded_pack_hit_count,
            "retired_pack_hit_count": self.retired_pack_hit_count,
            "environment_scope_violation_count": self.environment_scope_violation_count,
            "false_answerable": self.false_answerable,
            "false_insufficient": self.false_insufficient,
            "correct_insufficient": self.correct_insufficient,
            "repeated_run_disagreement_count": self.repeated_run_disagreement_count,
            "passed": self.passed,
            "pack_ids_observed": list(self.pack_ids_observed),
        }
        return data


def classify_pack_hits(state: ActivationStateManifest, pack_ids: Sequence[str], *,
                       environment: Optional[str] = None) -> Tuple[int, int, int, int]:
    """Count observed pack IDs whose governed status is *not* active-in-environment.

    Returns ``(inactive, superseded, retired, environment_scope_violation)``. A
    pack that is active but in a *different* environment counts as an
    environment-scope violation, not inactive. Read-only; never mutates state.
    """
    inactive = superseded = retired = env_violation = 0
    active_here = set(select_active_pack_ids(state, environment=environment))
    for pack_id in pack_ids:
        if pack_id in active_here:
            continue
        record = state.find(pack_id)
        if record is None:
            # An unknown pack id is treated as inactive (not governed-active).
            inactive += 1
            continue
        if record.status == PackLifecycleState.SUPERSEDED:
            superseded += 1
        elif record.status == PackLifecycleState.RETIRED:
            retired += 1
        elif record.status == PackLifecycleState.ACTIVE:
            # Active, but not in the requested environment.
            env_violation += 1
        else:
            inactive += 1
    return inactive, superseded, retired, env_violation


# ---------------------------------------------------------------------------
# Aggregate metrics (critical cases tracked separately)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonitoringMetrics:
    """Aggregate metrics over a set of case results, plus a critical-only view."""

    case_count: int = 0
    critical_case_count: int = 0
    passed_count: int = 0
    critical_passed_count: int = 0

    hit_at_k_rate: Optional[float] = None
    expected_source_recall: Optional[float] = None
    wrong_source_rate: Optional[float] = None
    off_topic_inclusion_rate: Optional[float] = None
    duplicate_interference_rate: Optional[float] = None

    selected_expected_source_recall: Optional[float] = None
    selected_forbidden_source_count: int = 0
    cited_expected_source_recall: Optional[float] = None

    forbidden_source_hit_count: int = 0  # gated (selected + cited)
    raw_forbidden_source_hit_count: int = 0
    unsupported_citation_count: int = 0
    uncited_factual_claim_count: int = 0

    inactive_pack_hit_count: int = 0
    superseded_pack_hit_count: int = 0
    retired_pack_hit_count: int = 0
    environment_scope_violation_count: int = 0

    unrelated_case_regression_count: int = 0
    false_answerable_count: int = 0
    false_insufficient_count: int = 0
    correct_insufficient_count: int = 0
    repeated_run_disagreement_count: int = 0

    @property
    def critical_failed_count(self) -> int:
        return self.critical_case_count - self.critical_passed_count

    def to_dict(self) -> dict:
        return {
            "case_count": self.case_count,
            "critical_case_count": self.critical_case_count,
            "passed_count": self.passed_count,
            "critical_passed_count": self.critical_passed_count,
            "critical_failed_count": self.critical_failed_count,
            "hit_at_k_rate": _round(self.hit_at_k_rate),
            "expected_source_recall": _round(self.expected_source_recall),
            "wrong_source_rate": _round(self.wrong_source_rate),
            "off_topic_inclusion_rate": _round(self.off_topic_inclusion_rate),
            "duplicate_interference_rate": _round(self.duplicate_interference_rate),
            "selected_expected_source_recall": _round(self.selected_expected_source_recall),
            "selected_forbidden_source_count": self.selected_forbidden_source_count,
            "cited_expected_source_recall": _round(self.cited_expected_source_recall),
            "forbidden_source_hit_count": self.forbidden_source_hit_count,
            "raw_forbidden_source_hit_count": self.raw_forbidden_source_hit_count,
            "unsupported_citation_count": self.unsupported_citation_count,
            "uncited_factual_claim_count": self.uncited_factual_claim_count,
            "inactive_pack_hit_count": self.inactive_pack_hit_count,
            "superseded_pack_hit_count": self.superseded_pack_hit_count,
            "retired_pack_hit_count": self.retired_pack_hit_count,
            "environment_scope_violation_count": self.environment_scope_violation_count,
            "unrelated_case_regression_count": self.unrelated_case_regression_count,
            "false_answerable_count": self.false_answerable_count,
            "false_insufficient_count": self.false_insufficient_count,
            "correct_insufficient_count": self.correct_insufficient_count,
            "repeated_run_disagreement_count": self.repeated_run_disagreement_count,
        }


def compute_metrics(results: Sequence[MonitoringCaseResult]) -> MonitoringMetrics:
    """Aggregate per-case results. Critical cases are counted separately."""
    critical = [r for r in results if r.is_critical]
    control = [r for r in results if r.control_case]

    def _hit_value(r: MonitoringCaseResult) -> Optional[float]:
        return None if r.hit is None else (1.0 if r.hit else 0.0)

    unrelated_regressions = sum(
        1 for r in control
        if r.selected_forbidden_source_count or r.gated_forbidden_hit_count
        or not r.passed)

    return MonitoringMetrics(
        case_count=len(results),
        critical_case_count=len(critical),
        passed_count=sum(1 for r in results if r.passed),
        critical_passed_count=sum(1 for r in critical if r.passed),
        hit_at_k_rate=_mean([v for r in results if (v := _hit_value(r)) is not None]),
        expected_source_recall=_mean(
            [r.expected_source_recall for r in results
             if r.expected_source_recall is not None]),
        wrong_source_rate=_mean(
            [r.wrong_source_rate for r in results
             if r.wrong_source_rate is not None]),
        off_topic_inclusion_rate=_mean(
            [1.0 if r.off_topic_inclusion else 0.0 for r in results]),
        duplicate_interference_rate=_mean(
            [r.duplicate_interference_rate for r in results]),
        selected_expected_source_recall=_mean(
            [r.selected_expected_source_recall for r in results
             if r.selected_expected_source_recall is not None]),
        selected_forbidden_source_count=sum(
            r.selected_forbidden_source_count for r in results),
        cited_expected_source_recall=_mean(
            [r.cited_expected_source_recall for r in results
             if r.cited_expected_source_recall is not None]),
        forbidden_source_hit_count=sum(r.gated_forbidden_hit_count for r in results),
        raw_forbidden_source_hit_count=sum(
            r.raw_forbidden_source_hit_count for r in results),
        unsupported_citation_count=sum(r.unsupported_citation_count for r in results),
        uncited_factual_claim_count=sum(r.uncited_factual_claim_count for r in results),
        inactive_pack_hit_count=sum(r.inactive_pack_hit_count for r in results),
        superseded_pack_hit_count=sum(r.superseded_pack_hit_count for r in results),
        retired_pack_hit_count=sum(r.retired_pack_hit_count for r in results),
        environment_scope_violation_count=sum(
            r.environment_scope_violation_count for r in results),
        unrelated_case_regression_count=unrelated_regressions,
        false_answerable_count=sum(1 for r in results if r.false_answerable),
        false_insufficient_count=sum(1 for r in results if r.false_insufficient),
        correct_insufficient_count=sum(1 for r in results if r.correct_insufficient),
        repeated_run_disagreement_count=sum(
            r.repeated_run_disagreement_count for r in results),
    )


# ---------------------------------------------------------------------------
# Active-pack snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActivePackSnapshot:
    """An immutable, fingerprinted view of the active-pack state at run time."""

    active_state_hash: str
    environment: str
    active_pack_ids: Tuple[str, ...]
    active_pack_versions: Tuple[str, ...]
    active_pack_fingerprints: Tuple[str, ...]
    source_ids: Tuple[str, ...]
    source_revisions: Tuple[str, ...]
    coexistence_policies: Tuple[str, ...]
    activation_audit_head_hash: str = ""
    captured_at: str = ""

    def _fp_payload(self) -> dict:
        return {
            "active_state_hash": self.active_state_hash,
            "environment": self.environment,
            "active_pack_ids": list(self.active_pack_ids),
            "active_pack_versions": list(self.active_pack_versions),
            "active_pack_fingerprints": list(self.active_pack_fingerprints),
            "source_ids": list(self.source_ids),
            "source_revisions": list(self.source_revisions),
            "coexistence_policies": list(self.coexistence_policies),
            "activation_audit_head_hash": self.activation_audit_head_hash,
        }

    @property
    def snapshot_id(self) -> str:
        return SNAPSHOT_FP_PREFIX + _sha256_hex(_canonical(self._fp_payload()))[:20]

    def to_dict(self) -> dict:
        return {"_record": "active_pack_snapshot", "snapshot_id": self.snapshot_id,
                **self._fp_payload(), "captured_at": self.captured_at}

    @classmethod
    def from_dict(cls, data: dict) -> "ActivePackSnapshot":
        def _tuple(key: str) -> Tuple[str, ...]:
            return tuple(str(v) for v in (data.get(key) or []))

        return cls(
            active_state_hash=str(data.get("active_state_hash", "")),
            environment=str(data.get("environment", "default")),
            active_pack_ids=_tuple("active_pack_ids"),
            active_pack_versions=_tuple("active_pack_versions"),
            active_pack_fingerprints=_tuple("active_pack_fingerprints"),
            source_ids=_tuple("source_ids"),
            source_revisions=_tuple("source_revisions"),
            coexistence_policies=_tuple("coexistence_policies"),
            activation_audit_head_hash=str(data.get("activation_audit_head_hash", "")),
            captured_at=str(data.get("captured_at", "")),
        )


def snapshot_from_state(state: ActivationStateManifest, *, environment: str = "default",
                        activation_audit_head_hash: str = "",
                        captured_at: Optional[str] = None) -> ActivePackSnapshot:
    """Capture an :class:`ActivePackSnapshot` from a governed activation manifest."""
    active = sorted(state.active(environment=environment), key=lambda r: r.pack_id)
    return ActivePackSnapshot(
        active_state_hash=state.state_hash,
        environment=environment,
        active_pack_ids=tuple(r.pack_id for r in active),
        active_pack_versions=tuple(r.pack_version for r in active),
        active_pack_fingerprints=tuple(r.pack_fingerprint for r in active),
        source_ids=tuple(r.source_id for r in active),
        source_revisions=tuple(r.source_revision for r in active),
        coexistence_policies=tuple(r.coexistence_policy.value for r in active),
        activation_audit_head_hash=activation_audit_head_hash,
        captured_at=captured_at if captured_at is not None else _utc_now_iso(),
    )


# ---------------------------------------------------------------------------
# Monitoring policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonitoringPolicy:
    """Deterministic thresholds that turn metric deltas into findings."""

    policy_id: str = "monitor-strict-v1"
    minimum_case_count: int = 4
    minimum_critical_case_count: int = 1
    allowed_regression_count: int = 0
    allowed_wrong_source_rate_delta: float = 0.05
    allowed_off_topic_delta: float = 0.0
    allowed_unrelated_case_regressions: int = 0
    allowed_unsupported_citations: int = 0
    allowed_inactive_or_superseded_hits: int = 0
    rank_worsened_tolerance: int = 0
    stability_run_count: int = 1
    minimum_repeated_run_agreement: int = 1
    evaluation_age_limit_days: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "_record": "monitoring_policy",
            "policy_id": self.policy_id,
            "minimum_case_count": self.minimum_case_count,
            "minimum_critical_case_count": self.minimum_critical_case_count,
            "allowed_regression_count": self.allowed_regression_count,
            "allowed_wrong_source_rate_delta": self.allowed_wrong_source_rate_delta,
            "allowed_off_topic_delta": self.allowed_off_topic_delta,
            "allowed_unrelated_case_regressions": self.allowed_unrelated_case_regressions,
            "allowed_unsupported_citations": self.allowed_unsupported_citations,
            "allowed_inactive_or_superseded_hits": self.allowed_inactive_or_superseded_hits,
            "rank_worsened_tolerance": self.rank_worsened_tolerance,
            "stability_run_count": self.stability_run_count,
            "minimum_repeated_run_agreement": self.minimum_repeated_run_agreement,
            "evaluation_age_limit_days": self.evaluation_age_limit_days,
        }

    @property
    def policy_fingerprint(self) -> str:
        return _sha256_hex(_canonical(self.to_dict()))[:16]

    @classmethod
    def from_dict(cls, data: dict) -> "MonitoringPolicy":
        return cls(
            policy_id=str(data.get("policy_id", "monitor-strict-v1")),
            minimum_case_count=int(data.get("minimum_case_count", 4)),
            minimum_critical_case_count=int(data.get("minimum_critical_case_count", 1)),
            allowed_regression_count=int(data.get("allowed_regression_count", 0)),
            allowed_wrong_source_rate_delta=float(
                data.get("allowed_wrong_source_rate_delta", 0.05)),
            allowed_off_topic_delta=float(data.get("allowed_off_topic_delta", 0.0)),
            allowed_unrelated_case_regressions=int(
                data.get("allowed_unrelated_case_regressions", 0)),
            allowed_unsupported_citations=int(
                data.get("allowed_unsupported_citations", 0)),
            allowed_inactive_or_superseded_hits=int(
                data.get("allowed_inactive_or_superseded_hits", 0)),
            rank_worsened_tolerance=int(data.get("rank_worsened_tolerance", 0)),
            stability_run_count=int(data.get("stability_run_count", 1)),
            minimum_repeated_run_agreement=int(
                data.get("minimum_repeated_run_agreement", 1)),
            evaluation_age_limit_days=(
                None if data.get("evaluation_age_limit_days") is None
                else int(data["evaluation_age_limit_days"])),
        )


DEFAULT_MONITORING_POLICY = MonitoringPolicy()


# ---------------------------------------------------------------------------
# Immutable baseline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonitoringBaseline:
    """An immutable, fingerprinted pre-activation (or other fixed) baseline."""

    baseline_id: str
    baseline_type: BaselineType
    active_state_hash: str
    snapshot_id: str
    pack_fingerprints: Tuple[str, ...]
    retrieval_config_fingerprint: str
    corpus_fingerprint: str
    metrics: MonitoringMetrics
    per_case: Tuple[dict, ...]
    evaluator_version: str = EVALUATOR_VERSION
    created_at: str = ""

    def _hash_payload(self) -> dict:
        return {
            "baseline_id": self.baseline_id,
            "baseline_type": self.baseline_type.value,
            "active_state_hash": self.active_state_hash,
            "snapshot_id": self.snapshot_id,
            "pack_fingerprints": list(self.pack_fingerprints),
            "retrieval_config_fingerprint": self.retrieval_config_fingerprint,
            "corpus_fingerprint": self.corpus_fingerprint,
            "metrics": self.metrics.to_dict(),
            "per_case": list(self.per_case),
            "evaluator_version": self.evaluator_version,
        }

    @property
    def baseline_hash(self) -> str:
        return BASELINE_HASH_PREFIX + _sha256_hex(_canonical(self._hash_payload()))[:20]

    def case_result(self, case_id: str) -> Optional[dict]:
        for row in self.per_case:
            if row.get("case_id") == case_id:
                return row
        return None

    def to_dict(self) -> dict:
        return {"_record": "monitoring_baseline", **self._hash_payload(),
                "baseline_hash": self.baseline_hash, "created_at": self.created_at}

    @classmethod
    def from_dict(cls, data: dict) -> "MonitoringBaseline":
        if data.get("_record") != "monitoring_baseline":
            raise ValueError("not a monitoring_baseline record")
        metrics_data = data.get("metrics") or {}
        return cls(
            baseline_id=str(data["baseline_id"]),
            baseline_type=BaselineType(str(data.get("baseline_type", "pre_activation"))),
            active_state_hash=str(data.get("active_state_hash", "")),
            snapshot_id=str(data.get("snapshot_id", "")),
            pack_fingerprints=tuple(
                str(v) for v in (data.get("pack_fingerprints") or [])),
            retrieval_config_fingerprint=str(data.get("retrieval_config_fingerprint", "")),
            corpus_fingerprint=str(data.get("corpus_fingerprint", "")),
            metrics=_metrics_from_dict(metrics_data),
            per_case=tuple(data.get("per_case") or []),
            evaluator_version=str(data.get("evaluator_version", EVALUATOR_VERSION)),
            created_at=str(data.get("created_at", "")),
        )


def _metrics_from_dict(data: dict) -> MonitoringMetrics:
    fields = {f for f in MonitoringMetrics.__dataclass_fields__}
    payload = {k: v for k, v in data.items() if k in fields}
    return MonitoringMetrics(**payload)


def create_baseline(*, baseline_id: str, baseline_type: BaselineType,
                    snapshot: ActivePackSnapshot, corpus_fingerprint: str,
                    retrieval_config_fingerprint: str,
                    results: Sequence[MonitoringCaseResult],
                    created_at: Optional[str] = None) -> MonitoringBaseline:
    """Construct an immutable baseline. Creation is explicit and separate from a run."""
    metrics = compute_metrics(results)
    return MonitoringBaseline(
        baseline_id=baseline_id,
        baseline_type=baseline_type,
        active_state_hash=snapshot.active_state_hash,
        snapshot_id=snapshot.snapshot_id,
        pack_fingerprints=tuple(snapshot.active_pack_fingerprints),
        retrieval_config_fingerprint=retrieval_config_fingerprint,
        corpus_fingerprint=corpus_fingerprint,
        metrics=metrics,
        per_case=tuple(r.to_dict() for r in results),
        evaluator_version=EVALUATOR_VERSION,
        created_at=created_at if created_at is not None else _utc_now_iso(),
    )


def load_baseline(path: str | Path) -> MonitoringBaseline:
    """Load a single immutable baseline record from a JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return MonitoringBaseline.from_dict(data)


def write_baseline(baseline: MonitoringBaseline, path: str | Path) -> Path:
    """Write a baseline to ``path`` — fails closed if a baseline already exists.

    Baselines are immutable: a new baseline must receive a new identity and a new
    path. This writer refuses to silently overwrite an existing baseline file.
    """
    target = Path(path)
    if target.exists():
        raise FileExistsError(
            f"baseline already exists at {target}; baselines are immutable — "
            "use a new path/identity")
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(target, json.dumps(baseline.to_dict(), indent=2,
                                          sort_keys=True))
    return target


def baseline_compatible(baseline: MonitoringBaseline, *, corpus_fingerprint: str,
                        retrieval_config_fingerprint: str,
                        evaluator_version: str = EVALUATOR_VERSION) -> bool:
    """A baseline is comparable only when corpus, retrieval config and evaluator match.

    The active state is *expected* to differ (the baseline is pre-activation), so
    it is deliberately **not** part of the compatibility check.
    """
    return (baseline.corpus_fingerprint == corpus_fingerprint
            and baseline.retrieval_config_fingerprint == retrieval_config_fingerprint
            and baseline.evaluator_version == evaluator_version)


# ---------------------------------------------------------------------------
# Metric deltas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonitoringMetricDelta:
    metric: str
    baseline_value: Optional[float]
    current_value: Optional[float]
    absolute_delta: Optional[float]
    relative_delta: Optional[float]
    improved: bool
    regressed: bool

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "baseline_value": _round(self.baseline_value),
            "current_value": _round(self.current_value),
            "absolute_delta": _round(self.absolute_delta),
            "relative_delta": _round(self.relative_delta),
            "improved": self.improved,
            "regressed": self.regressed,
        }


# Metrics where a *higher* value is better (recall/hit). All others: lower better.
_HIGHER_IS_BETTER = frozenset({
    "hit_at_k_rate", "expected_source_recall", "selected_expected_source_recall",
    "cited_expected_source_recall", "correct_insufficient_count",
})


def compute_deltas(baseline: MonitoringMetrics,
                   current: MonitoringMetrics) -> List[MonitoringMetricDelta]:
    """Deterministic per-metric deltas (baseline vs current)."""
    base = baseline.to_dict()
    curr = current.to_dict()
    deltas: List[MonitoringMetricDelta] = []
    for metric in sorted(curr):
        if metric in ("critical_failed_count",):
            continue
        b = base.get(metric)
        c = curr.get(metric)
        if not isinstance(b, (int, float)) or not isinstance(c, (int, float)):
            continue
        absolute = c - b
        relative = (absolute / abs(b)) if b not in (0, 0.0) else None
        higher_better = metric in _HIGHER_IS_BETTER
        improved = (absolute > 0) if higher_better else (absolute < 0)
        regressed = (absolute < 0) if higher_better else (absolute > 0)
        deltas.append(MonitoringMetricDelta(
            metric=metric, baseline_value=float(b), current_value=float(c),
            absolute_delta=float(absolute), relative_delta=relative,
            improved=bool(improved and absolute != 0),
            regressed=bool(regressed and absolute != 0)))
    return deltas


# ---------------------------------------------------------------------------
# Regression findings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegressionFinding:
    finding_code: RegressionFindingCode
    severity: MonitoringSeverity
    case_ids: Tuple[str, ...] = ()
    baseline_value: Optional[float] = None
    current_value: Optional[float] = None
    delta: Optional[float] = None
    candidate_pack_ids: Tuple[str, ...] = ()
    confidence: ConfidenceBand = ConfidenceBand.MEDIUM
    diagnostic_note: str = ""
    recommended_human_action: str = ""

    @property
    def is_critical(self) -> bool:
        return (self.severity == MonitoringSeverity.CRITICAL
                or self.finding_code in _CRITICAL_FINDING_CODES)

    def to_dict(self) -> dict:
        return {
            "_record": "regression_finding",
            "finding_code": self.finding_code.value,
            "severity": self.severity.value,
            "case_ids": list(self.case_ids),
            "baseline_value": _round(self.baseline_value),
            "current_value": _round(self.current_value),
            "delta": _round(self.delta),
            "candidate_pack_ids": list(self.candidate_pack_ids),
            "confidence": self.confidence.value,
            "diagnostic_note": self.diagnostic_note,
            "recommended_human_action": self.recommended_human_action,
        }


def _baseline_case_passed(baseline: MonitoringBaseline, case_id: str) -> Optional[bool]:
    row = baseline.case_result(case_id)
    return None if row is None else bool(row.get("passed", True))


def detect_regressions(*, baseline: MonitoringBaseline,
                       results: Sequence[MonitoringCaseResult],
                       snapshot: ActivePackSnapshot,
                       policy: MonitoringPolicy = DEFAULT_MONITORING_POLICY,
                       corpus_fingerprint: str,
                       retrieval_config_fingerprint: str) -> List[RegressionFinding]:
    """Compare current results to an immutable baseline and emit regression findings.

    Critical-case failures are detected independently of aggregate movement; an
    aggregate improvement can never suppress a critical failure.
    """
    findings: List[RegressionFinding] = []
    candidate_packs = tuple(snapshot.active_pack_ids)

    # Baseline compatibility (corpus / retrieval config / evaluator).
    if not baseline_compatible(
            baseline, corpus_fingerprint=corpus_fingerprint,
            retrieval_config_fingerprint=retrieval_config_fingerprint):
        findings.append(RegressionFinding(
            finding_code=RegressionFindingCode.BASELINE_INCOMPATIBLE,
            severity=MonitoringSeverity.HIGH,
            confidence=ConfidenceBand.HIGH,
            diagnostic_note=(
                "baseline corpus/retrieval-config/evaluator differs from the "
                "current run; comparison is not valid"),
            recommended_human_action="re-create a baseline for the current configuration"))
        return findings

    current = compute_metrics(results)
    baseline_metrics = baseline.metrics

    # Sample sufficiency.
    if current.case_count < policy.minimum_case_count:
        findings.append(RegressionFinding(
            finding_code=RegressionFindingCode.INSUFFICIENT_SAMPLE,
            severity=MonitoringSeverity.MEDIUM,
            current_value=float(current.case_count),
            baseline_value=float(policy.minimum_case_count),
            confidence=ConfidenceBand.HIGH,
            diagnostic_note="too few cases to support a strong recommendation",
            recommended_human_action="expand the monitoring corpus"))

    # -- Per-case safety + expected-content regressions ----------------------
    for r in results:
        base_passed = _baseline_case_passed(baseline, r.case_id)

        if r.gated_forbidden_hit_count:
            findings.append(_finding(
                RegressionFindingCode.FORBIDDEN_SOURCE_INTRODUCED,
                MonitoringSeverity.CRITICAL, r, candidate_packs,
                note="a forbidden source reached selected/cited evidence",
                action="investigate source isolation; rollback eligible"))

        if r.inactive_pack_hit_count:
            findings.append(_finding(
                RegressionFindingCode.INACTIVE_PACK_RETRIEVED,
                MonitoringSeverity.CRITICAL, r, candidate_packs,
                note="retrieval surfaced a non-active pack",
                action="verify active-set selection; rollback eligible"))
        if r.superseded_pack_hit_count:
            findings.append(_finding(
                RegressionFindingCode.SUPERSEDED_PACK_RETRIEVED,
                MonitoringSeverity.CRITICAL, r, candidate_packs,
                note="retrieval surfaced a superseded pack",
                action="verify supersession; rollback eligible"))
        if r.retired_pack_hit_count:
            findings.append(_finding(
                RegressionFindingCode.RETIRED_PACK_RETRIEVED,
                MonitoringSeverity.CRITICAL, r, candidate_packs,
                note="retrieval surfaced a retired pack",
                action="verify retirement; rollback eligible"))
        if r.environment_scope_violation_count:
            findings.append(_finding(
                RegressionFindingCode.ENVIRONMENT_SCOPE_VIOLATION,
                MonitoringSeverity.CRITICAL, r, candidate_packs,
                note="a pack active in another environment was retrieved here",
                action="verify environment scoping"))

        if r.unsupported_citation_count > policy.allowed_unsupported_citations:
            findings.append(_finding(
                RegressionFindingCode.UNSUPPORTED_CITATION_INTRODUCED,
                MonitoringSeverity.HIGH, r, candidate_packs,
                note="a citation references evidence not in the selected set",
                action="inspect citation integrity"))
        if r.uncited_factual_claim_count:
            findings.append(_finding(
                RegressionFindingCode.UNCITED_CLAIM_INTRODUCED,
                MonitoringSeverity.HIGH, r, candidate_packs,
                note="a factual claim was made without a citation",
                action="inspect grounding/guard"))
        if (r.citation_lineage_accuracy is not None
                and r.citation_lineage_accuracy < 1.0):
            findings.append(_finding(
                RegressionFindingCode.CITATION_LINEAGE_DEGRADED,
                MonitoringSeverity.MEDIUM, r, candidate_packs,
                current=r.citation_lineage_accuracy,
                note="citation lineage accuracy dropped below 1.0"))

        if r.false_answerable:
            findings.append(_finding(
                RegressionFindingCode.FALSE_ANSWERABLE_INTRODUCED,
                MonitoringSeverity.HIGH, r, candidate_packs,
                note="an insufficient-evidence case produced a grounded answer"))
        if r.false_insufficient:
            findings.append(_finding(
                RegressionFindingCode.FALSE_INSUFFICIENT_INTRODUCED,
                MonitoringSeverity.MEDIUM, r, candidate_packs,
                note="an answerable case was reported as insufficient"))

        # Expected-content drops (only when the case expected something and the
        # baseline had it).
        base_row = baseline.case_result(r.case_id)
        if base_row is not None:
            base_recall = base_row.get("expected_source_recall")
            if (isinstance(base_recall, (int, float))
                    and r.expected_source_recall is not None
                    and r.expected_source_recall < base_recall):
                findings.append(_finding(
                    RegressionFindingCode.EXPECTED_SOURCE_DROPPED,
                    MonitoringSeverity.HIGH, r, candidate_packs,
                    baseline=base_recall, current=r.expected_source_recall,
                    note="expected-source recall fell below baseline"))
            base_chunk = base_row.get("expected_chunk_recall")
            if (isinstance(base_chunk, (int, float))
                    and r.expected_chunk_recall is not None
                    and r.expected_chunk_recall < base_chunk):
                findings.append(_finding(
                    RegressionFindingCode.EXPECTED_CHUNK_DROPPED,
                    MonitoringSeverity.MEDIUM, r, candidate_packs,
                    baseline=base_chunk, current=r.expected_chunk_recall,
                    note="expected-chunk recall fell below baseline"))
            base_rank = base_row.get("first_expected_rank")
            if (isinstance(base_rank, int) and r.first_expected_rank is not None
                    and r.first_expected_rank
                    > base_rank + policy.rank_worsened_tolerance):
                findings.append(_finding(
                    RegressionFindingCode.EXPECTED_SOURCE_RANK_WORSENED,
                    MonitoringSeverity.MEDIUM, r, candidate_packs,
                    baseline=float(base_rank), current=float(r.first_expected_rank),
                    note="expected source now ranks lower than baseline"))

        # Control / unrelated regressions.
        if r.control_case and base_passed and not r.passed:
            findings.append(_finding(
                RegressionFindingCode.UNRELATED_CASE_REGRESSED,
                MonitoringSeverity.HIGH, r, candidate_packs,
                note="an unrelated control case now regresses"))

        if (r.duplicate_interference_rate
                and base_row is not None
                and r.duplicate_interference_rate
                > (base_row.get("duplicate_interference_rate") or 0.0)):
            findings.append(_finding(
                RegressionFindingCode.DUPLICATE_INTERFERENCE_INCREASED,
                MonitoringSeverity.LOW, r, candidate_packs,
                current=r.duplicate_interference_rate,
                note="duplicate-chunk interference increased over baseline"))

        if not r.stable:
            findings.append(_finding(
                RegressionFindingCode.INSTABILITY_DETECTED,
                MonitoringSeverity.MEDIUM, r, candidate_packs,
                confidence=ConfidenceBand.LOW,
                note="repeated runs disagreed for this case"))

        if r.is_critical and not r.passed:
            findings.append(_finding(
                RegressionFindingCode.CRITICAL_CASE_FAILED,
                MonitoringSeverity.CRITICAL, r, candidate_packs,
                confidence=ConfidenceBand.HIGH,
                note="a critical case failed under the active set",
                action="rollback eligible if a validated known-good target exists"))

    # -- Aggregate wrong-source movement ------------------------------------
    if (baseline_metrics.wrong_source_rate is not None
            and current.wrong_source_rate is not None):
        delta = current.wrong_source_rate - baseline_metrics.wrong_source_rate
        if delta > policy.allowed_wrong_source_rate_delta:
            findings.append(RegressionFinding(
                finding_code=RegressionFindingCode.WRONG_SOURCE_RATE_INCREASED,
                severity=MonitoringSeverity.MEDIUM,
                baseline_value=baseline_metrics.wrong_source_rate,
                current_value=current.wrong_source_rate, delta=delta,
                candidate_pack_ids=candidate_packs,
                confidence=ConfidenceBand.MEDIUM,
                diagnostic_note="aggregate wrong-source rate rose beyond tolerance",
                recommended_human_action="inspect ranking/source selection"))

    # -- Aggregate-improved-but-critical-failed safety override -------------
    critical_failed = current.critical_failed_count > 0
    aggregate_improved = (
        baseline_metrics.expected_source_recall is not None
        and current.expected_source_recall is not None
        and current.expected_source_recall >= baseline_metrics.expected_source_recall)
    if critical_failed and aggregate_improved:
        findings.append(RegressionFinding(
            finding_code=RegressionFindingCode.AGGREGATE_IMPROVED_BUT_CRITICAL_FAILED,
            severity=MonitoringSeverity.CRITICAL,
            candidate_pack_ids=candidate_packs,
            confidence=ConfidenceBand.HIGH,
            diagnostic_note=(
                "aggregate recall held or improved while a critical case failed; "
                "aggregate improvement must not mask a critical regression"),
            recommended_human_action="treat as a critical regression regardless of aggregate"))

    findings.sort(key=lambda f: (_SEVERITY_RANK[f.severity], f.finding_code.value,
                                 tuple(f.case_ids)))
    return findings


def _finding(code: RegressionFindingCode, severity: MonitoringSeverity,
             result: MonitoringCaseResult, candidate_packs: Tuple[str, ...], *,
             baseline: Optional[float] = None, current: Optional[float] = None,
             confidence: ConfidenceBand = ConfidenceBand.MEDIUM,
             note: str = "", action: str = "") -> RegressionFinding:
    return RegressionFinding(
        finding_code=code, severity=severity, case_ids=(result.case_id,),
        baseline_value=baseline, current_value=current,
        delta=(None if baseline is None or current is None else current - baseline),
        candidate_pack_ids=candidate_packs, confidence=confidence,
        diagnostic_note=note, recommended_human_action=action)


# ---------------------------------------------------------------------------
# Conservative attribution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttributionResult:
    attribution: AttributionClass
    candidate_pack_ids: Tuple[str, ...]
    confidence: ConfidenceBand
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "_record": "attribution_result",
            "attribution": self.attribution.value,
            "candidate_pack_ids": list(self.candidate_pack_ids),
            "confidence": self.confidence.value,
            "note": self.note,
        }


def attribute_regression(*, regression_with_pack: bool, regression_without_pack: bool,
                         retrieval_config_changed: bool, corpus_changed: bool,
                         evaluator_changed: bool, stable: bool,
                         candidate_pack_ids: Sequence[str]) -> AttributionResult:
    """Classify the cause of a regression conservatively.

    ``pack_specific`` is asserted **only** when the regression appears with the
    pack enabled, disappears with it disabled, configuration and corpus are
    unchanged, and repeated runs are stable. Anything else is classified more
    conservatively — never as pack-specific by assumption.
    """
    packs = tuple(candidate_pack_ids)
    if corpus_changed:
        return AttributionResult(AttributionClass.CORPUS_CHANGE, packs,
                                 ConfidenceBand.HIGH,
                                 "corpus changed; baseline comparison is not valid")
    if evaluator_changed:
        return AttributionResult(AttributionClass.EVALUATOR_CHANGE, packs,
                                 ConfidenceBand.HIGH, "evaluator version changed")
    if retrieval_config_changed:
        return AttributionResult(AttributionClass.RETRIEVAL_CONFIGURATION, packs,
                                 ConfidenceBand.MEDIUM,
                                 "retrieval configuration changed; cannot isolate a pack")
    if not stable:
        return AttributionResult(AttributionClass.UNSTABLE_RESULT, packs,
                                 ConfidenceBand.LOW,
                                 "repeated runs disagreed; cause undetermined")
    if regression_with_pack and not regression_without_pack:
        return AttributionResult(AttributionClass.PACK_SPECIFIC, packs,
                                 ConfidenceBand.HIGH,
                                 "regression present with pack, absent without it")
    if regression_with_pack and regression_without_pack:
        return AttributionResult(AttributionClass.ACTIVE_SET_INTERACTION, packs,
                                 ConfidenceBand.MEDIUM,
                                 "regression persists with the pack disabled; "
                                 "likely an active-set interaction")
    return AttributionResult(AttributionClass.UNDETERMINED, packs,
                             ConfidenceBand.LOW, "cause undetermined")


# ---------------------------------------------------------------------------
# Deterministic recommendations (advisory only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonitoringRecommendation:
    recommendation: MonitoringRecommendationCode
    rationale_codes: Tuple[str, ...]
    confidence: ConfidenceBand
    affected_cases: Tuple[str, ...] = ()
    candidate_pack_ids: Tuple[str, ...] = ()
    rollback_target: str = ""
    required_human_approval: str = ""
    generated_at: str = ""

    @property
    def advisory_only(self) -> bool:
        return True

    def to_dict(self) -> dict:
        return {
            "_record": "monitoring_recommendation",
            "recommendation": self.recommendation.value,
            "advisory_only": True,
            "rationale_codes": list(self.rationale_codes),
            "confidence": self.confidence.value,
            "affected_cases": list(self.affected_cases),
            "candidate_pack_ids": list(self.candidate_pack_ids),
            "rollback_target": self.rollback_target,
            "required_human_approval": self.required_human_approval,
            "generated_at": self.generated_at,
        }


def recommend(*, findings: Sequence[RegressionFinding], metrics: MonitoringMetrics,
              policy: MonitoringPolicy = DEFAULT_MONITORING_POLICY,
              snapshot: Optional[ActivePackSnapshot] = None,
              rollback_target: str = "",
              pack_attribution: Optional[AttributionResult] = None,
              generated_at: Optional[str] = None) -> MonitoringRecommendation:
    """Map findings + metrics to a single closed-set, advisory recommendation."""
    candidate_packs = tuple(snapshot.active_pack_ids) if snapshot else ()
    now = generated_at if generated_at is not None else _utc_now_iso()
    codes = {f.finding_code for f in findings}
    affected = tuple(sorted({cid for f in findings for cid in f.case_ids}))

    def _result(rec: MonitoringRecommendationCode, rationale: Sequence[str],
                confidence: ConfidenceBand, approval: str = "",
                target: str = "") -> MonitoringRecommendation:
        return MonitoringRecommendation(
            recommendation=rec, rationale_codes=tuple(sorted(set(rationale))),
            confidence=confidence, affected_cases=affected,
            candidate_pack_ids=candidate_packs, rollback_target=target,
            required_human_approval=approval, generated_at=now)

    # Invalid run / incompatible baseline -> cannot recommend.
    if (RegressionFindingCode.MONITORING_RUN_INVALID in codes
            or RegressionFindingCode.BASELINE_INCOMPATIBLE in codes):
        return _result(
            MonitoringRecommendationCode.INSUFFICIENT_EVIDENCE_TO_RECOMMEND,
            [RegressionFindingCode.BASELINE_INCOMPATIBLE.value], ConfidenceBand.HIGH)

    # Insufficient sample blocks any strong keep/rollback recommendation.
    if RegressionFindingCode.INSUFFICIENT_SAMPLE in codes:
        return _result(
            MonitoringRecommendationCode.INSUFFICIENT_EVIDENCE_TO_RECOMMEND,
            [RegressionFindingCode.INSUFFICIENT_SAMPLE.value], ConfidenceBand.HIGH)

    critical = [f for f in findings if f.is_critical]
    pack_specific = (pack_attribution is not None
                     and pack_attribution.attribution == AttributionClass.PACK_SPECIFIC)

    # Critical regression with a validated rollback target + strong pack
    # attribution -> rollback recommendation (still advisory).
    if critical:
        if rollback_target and pack_specific:
            return _result(
                MonitoringRecommendationCode.ROLLBACK_RECOMMENDED,
                [f.finding_code.value for f in critical],
                ConfidenceBand.HIGH, approval="rollback_approval", target=rollback_target)
        if pack_specific:
            return _result(
                MonitoringRecommendationCode.DEACTIVATE_RECOMMENDED,
                [f.finding_code.value for f in critical],
                ConfidenceBand.MEDIUM, approval="deactivation_approval")
        return _result(
            MonitoringRecommendationCode.INVESTIGATE,
            [f.finding_code.value for f in critical], ConfidenceBand.MEDIUM,
            approval="investigation")

    # Repeated failed activations / unresolved provenance defects.
    if RegressionFindingCode.BASELINE_INCOMPATIBLE in codes:
        return _result(MonitoringRecommendationCode.INVESTIGATE,
                       [RegressionFindingCode.BASELINE_INCOMPATIBLE.value],
                       ConfidenceBand.MEDIUM)

    high = [f for f in findings if f.severity == MonitoringSeverity.HIGH]
    instability = RegressionFindingCode.INSTABILITY_DETECTED in codes
    undetermined = (pack_attribution is not None
                    and pack_attribution.attribution in (
                        AttributionClass.UNDETERMINED,
                        AttributionClass.UNSTABLE_RESULT,
                        AttributionClass.ACTIVE_SET_INTERACTION))

    if high:
        if pack_specific:
            return _result(
                MonitoringRecommendationCode.DEACTIVATE_RECOMMENDED,
                [f.finding_code.value for f in high], ConfidenceBand.MEDIUM,
                approval="deactivation_approval")
        return _result(
            MonitoringRecommendationCode.INVESTIGATE,
            [f.finding_code.value for f in high], ConfidenceBand.MEDIUM,
            approval="investigation")

    if instability or undetermined:
        return _result(MonitoringRecommendationCode.INVESTIGATE,
                       ["unstable_or_undetermined"], ConfidenceBand.LOW,
                       approval="investigation")

    if findings:
        # Only minor (low/medium, non-safety) degradation remains.
        return _result(MonitoringRecommendationCode.KEEP_ACTIVE_WITH_WATCH,
                       [f.finding_code.value for f in findings], ConfidenceBand.MEDIUM)

    return _result(MonitoringRecommendationCode.KEEP_ACTIVE,
                   ["no_regressions_detected"], ConfidenceBand.HIGH)


# ---------------------------------------------------------------------------
# Monitoring run + report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonitoringRun:
    """A bound monitoring run: snapshot + baseline + corpus + policy + outcome."""

    monitoring_run_id: str
    snapshot: ActivePackSnapshot
    baseline_hash: str
    corpus_fingerprint: str
    retrieval_config_fingerprint: str
    policy_id: str
    policy_fingerprint: str
    metrics: MonitoringMetrics
    deltas: Tuple[MonitoringMetricDelta, ...]
    findings: Tuple[RegressionFinding, ...]
    recommendation: MonitoringRecommendation
    evaluator_version: str = EVALUATOR_VERSION
    created_at: str = ""

    @property
    def critical_findings(self) -> Tuple[RegressionFinding, ...]:
        return tuple(f for f in self.findings if f.is_critical)

    def is_valid_for(self, *, active_state_hash: str, corpus_fingerprint: str,
                     retrieval_config_fingerprint: str,
                     policy_fingerprint: str) -> bool:
        """A stored run is valid only while its bound identity still holds."""
        return (self.snapshot.active_state_hash == active_state_hash
                and self.corpus_fingerprint == corpus_fingerprint
                and self.retrieval_config_fingerprint == retrieval_config_fingerprint
                and self.policy_fingerprint == policy_fingerprint)

    def _hash_payload(self) -> dict:
        return {
            "snapshot_id": self.snapshot.snapshot_id,
            "active_state_hash": self.snapshot.active_state_hash,
            "baseline_hash": self.baseline_hash,
            "corpus_fingerprint": self.corpus_fingerprint,
            "retrieval_config_fingerprint": self.retrieval_config_fingerprint,
            "policy_fingerprint": self.policy_fingerprint,
            "metrics": self.metrics.to_dict(),
            "findings": [f.to_dict() for f in self.findings],
            "recommendation": self.recommendation.recommendation.value,
            "evaluator_version": self.evaluator_version,
        }

    @property
    def record_hash(self) -> str:
        return RECORD_HASH_PREFIX + _sha256_hex(_canonical(self._hash_payload()))[:20]

    def to_dict(self) -> dict:
        return {
            "_record": "active_pack_monitoring_run",
            "monitoring_run_id": self.monitoring_run_id,
            "snapshot": self.snapshot.to_dict(),
            "baseline_hash": self.baseline_hash,
            "corpus_fingerprint": self.corpus_fingerprint,
            "retrieval_config_fingerprint": self.retrieval_config_fingerprint,
            "policy_id": self.policy_id,
            "policy_fingerprint": self.policy_fingerprint,
            "metrics": self.metrics.to_dict(),
            "deltas": [d.to_dict() for d in self.deltas],
            "findings": [f.to_dict() for f in self.findings],
            "recommendation": self.recommendation.to_dict(),
            "evaluator_version": self.evaluator_version,
            "created_at": self.created_at,
            "record_hash": self.record_hash,
        }


def run_monitoring(*, baseline: MonitoringBaseline, snapshot: ActivePackSnapshot,
                   results: Sequence[MonitoringCaseResult],
                   corpus_fingerprint: str, retrieval_config_fingerprint: str,
                   policy: MonitoringPolicy = DEFAULT_MONITORING_POLICY,
                   rollback_target: str = "",
                   pack_attribution: Optional[AttributionResult] = None,
                   created_at: Optional[str] = None) -> MonitoringRun:
    """Pure orchestrator: metrics -> deltas -> findings -> recommendation."""
    now = created_at if created_at is not None else _utc_now_iso()
    metrics = compute_metrics(results)
    deltas = compute_deltas(baseline.metrics, metrics)
    findings = detect_regressions(
        baseline=baseline, results=results, snapshot=snapshot, policy=policy,
        corpus_fingerprint=corpus_fingerprint,
        retrieval_config_fingerprint=retrieval_config_fingerprint)
    recommendation = recommend(
        findings=findings, metrics=metrics, policy=policy, snapshot=snapshot,
        rollback_target=rollback_target, pack_attribution=pack_attribution,
        generated_at=now)
    run_id = RUN_HASH_PREFIX + _sha256_hex(_canonical({
        "snapshot": snapshot.snapshot_id,
        "baseline": baseline.baseline_hash,
        "corpus": corpus_fingerprint,
        "policy": policy.policy_fingerprint,
        "created_at": now,
    }))[:16]
    return MonitoringRun(
        monitoring_run_id=run_id, snapshot=snapshot,
        baseline_hash=baseline.baseline_hash, corpus_fingerprint=corpus_fingerprint,
        retrieval_config_fingerprint=retrieval_config_fingerprint,
        policy_id=policy.policy_id, policy_fingerprint=policy.policy_fingerprint,
        metrics=metrics, deltas=tuple(deltas), findings=tuple(findings),
        recommendation=recommendation, evaluator_version=EVALUATOR_VERSION,
        created_at=now)


# ---------------------------------------------------------------------------
# Append-only monitoring history
# ---------------------------------------------------------------------------


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def append_monitoring_history(run: MonitoringRun,
                              path: str | Path = DEFAULT_HISTORY_PATH) -> Path:
    """Append a monitoring run to the append-only history file.

    History is never rewritten silently — this only ever appends one JSONL line.
    Only IDs, hashes, bounded metrics and findings are stored (no retrieved text).
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = _canonical(run.to_dict())
    with open(target, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    return target


def load_monitoring_history(path: str | Path = DEFAULT_HISTORY_PATH) -> List[dict]:
    """Load the append-only monitoring history (missing file -> empty list)."""
    target = Path(path)
    if not target.exists():
        return []
    rows: List[dict] = []
    for raw in target.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Deterministic markdown renderers (critical findings first; advisory labelled)
# ---------------------------------------------------------------------------


def render_monitoring_markdown(run: MonitoringRun) -> str:
    lines = [
        "# Active-pack monitoring report (v7.1; read-only)",
        "",
        f"- Active-state fingerprint: `{run.snapshot.snapshot_id}`",
        f"- Active-state hash: `{run.snapshot.active_state_hash}`",
        f"- Baseline: `{run.baseline_hash}`",
        f"- Corpus fingerprint: `{run.corpus_fingerprint}`",
        f"- Retrieval config: `{run.retrieval_config_fingerprint}`",
        f"- Policy: `{run.policy_id}` (`{run.policy_fingerprint}`)",
        f"- Cases: {run.metrics.case_count} "
        f"(critical: {run.metrics.critical_case_count}, "
        f"critical failed: {run.metrics.critical_failed_count})",
        "",
        f"## Recommendation (ADVISORY ONLY): "
        f"**{run.recommendation.recommendation.value}**",
        f"- Confidence: {run.recommendation.confidence.value}",
        f"- Required human approval: "
        f"{run.recommendation.required_human_approval or 'none'}",
    ]
    if run.recommendation.rollback_target:
        lines.append(f"- Rollback target: `{run.recommendation.rollback_target}`")
    lines.append(
        "- This is a recommendation, not an action; no lifecycle change was made.")
    lines.append("")

    critical = run.critical_findings
    lines.append("## Critical findings")
    if critical:
        lines.append("| Code | Cases | Confidence | Note |")
        lines.append("| --- | --- | --- | --- |")
        for f in critical:
            lines.append(
                f"| {f.finding_code.value} | {', '.join(f.case_ids) or '-'} | "
                f"{f.confidence.value} | {f.diagnostic_note} |")
    else:
        lines.append("_No critical findings._")
    lines.append("")

    other = [f for f in run.findings if not f.is_critical]
    lines.append("## Other findings")
    if other:
        lines.append("| Code | Severity | Cases | Note |")
        lines.append("| --- | --- | --- | --- |")
        for f in other:
            lines.append(
                f"| {f.finding_code.value} | {f.severity.value} | "
                f"{', '.join(f.case_ids) or '-'} | {f.diagnostic_note} |")
    else:
        lines.append("_No other findings._")
    return "\n".join(lines)


def render_history_markdown(history: Sequence[dict]) -> str:
    lines = ["# Active-pack monitoring history", ""]
    if not history:
        lines.append("_No monitoring history._")
        return "\n".join(lines)
    lines.append("| Timestamp | Run | Active state | Recommendation | Critical |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in history:
        rec = (row.get("recommendation") or {}).get("recommendation", "")
        metrics = row.get("metrics") or {}
        lines.append(
            f"| {row.get('created_at', '')} | `{row.get('monitoring_run_id', '')}` | "
            f"`{row.get('snapshot', {}).get('active_state_hash', '')}` | {rec} | "
            f"{metrics.get('critical_failed_count', 0)} |")
    return "\n".join(lines)
