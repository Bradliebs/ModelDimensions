"""Dataset card / metadata access for Hugging Face imports (v1.9).

A Hugging Face dataset being *free to download* says nothing about whether it is
*safe to trust*. Before any rows are sampled, this module gathers what the
dataset itself declares about its provenance — its licence, tags, task
categories, size, citation, and revision — from its dataset card. That metadata
is what the importer uses to decide whether an import is allowed at all.

Two access paths, in order of preference:

* **Offline fixture.** A local JSON card (``card_path``) is read directly. This
  is how tests and the bundled demos run — no network is ever touched.
* **Network (opt-in only).** When ``allow_network`` is explicitly set, a best
  effort lookup through ``huggingface_hub`` is attempted. Any failure (offline,
  package missing, unknown dataset) degrades gracefully to a "card absent"
  result rather than raising, so a missing network never crashes a caller.

The default is offline: with neither a fixture nor explicit network opt-in, the
result is an honest "no card, licence unknown" — which the importer treats as
untrusted.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Licence strings that mean "no usable licence was declared".
_UNKNOWN_LICENCE_VALUES = {"", "unknown", "unlicensed", "other", "none", "null"}


def licence_is_known(licence: Optional[str]) -> bool:
    """True only for a concrete, declared licence string."""
    if licence is None:
        return False
    return licence.strip().lower() not in _UNKNOWN_LICENCE_VALUES


@dataclass
class HFDatasetMetadata:
    """What a dataset declares about itself, normalised across access paths."""

    dataset_id: str
    card_present: bool = False
    license: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    task_categories: List[str] = field(default_factory=list)
    size_info: dict = field(default_factory=dict)
    citation: Optional[str] = None
    revision: Optional[str] = None
    retrieved_at: str = field(default_factory=_utc_now)
    source: str = "absent"  # "fixture" | "network" | "absent"
    note: Optional[str] = None

    @property
    def license_known(self) -> bool:
        return licence_is_known(self.license)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["license_known"] = self.license_known
        return data


def read_card_file(path: str | Path) -> HFDatasetMetadata:
    """Read a local dataset-card JSON fixture into normalised metadata.

    The fixture is expected to be a mapping; missing keys degrade to empty
    defaults so a partial card still loads.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"dataset card {path} must be a JSON object")
    return HFDatasetMetadata(
        dataset_id=str(data.get("dataset_id") or path.stem),
        card_present=bool(data.get("card_present", True)),
        license=_opt_str(data.get("license")),
        tags=_str_list(data.get("tags")),
        task_categories=_str_list(data.get("task_categories")),
        size_info=dict(data.get("size_info") or {}),
        citation=_opt_str(data.get("citation")),
        revision=_opt_str(data.get("revision")),
        source="fixture",
    )


def _fetch_via_hub(dataset_id: str,
                   revision: Optional[str]) -> Optional[HFDatasetMetadata]:
    """Best-effort metadata via ``huggingface_hub``; ``None`` on any failure."""
    try:  # pragma: no cover - network path is never exercised in tests
        from huggingface_hub import dataset_info  # type: ignore

        info = dataset_info(dataset_id, revision=revision)
        card = dict(getattr(info, "card_data", None) or {})
        return HFDatasetMetadata(
            dataset_id=dataset_id,
            card_present=bool(card),
            license=_opt_str(card.get("license")),
            tags=_str_list(getattr(info, "tags", None) or card.get("tags")),
            task_categories=_str_list(card.get("task_categories")),
            size_info={},
            citation=_opt_str(card.get("citation")),
            revision=_opt_str(getattr(info, "sha", None) or revision),
            source="network",
        )
    except Exception as exc:  # pragma: no cover - defensive, offline-safe
        return HFDatasetMetadata(
            dataset_id=dataset_id,
            card_present=False,
            source="absent",
            note=f"metadata lookup failed: {exc.__class__.__name__}",
        )


def fetch_metadata(dataset_id: str, *,
                   card_path: Optional[str | Path] = None,
                   revision: Optional[str] = None,
                   allow_network: bool = False) -> HFDatasetMetadata:
    """Resolve a dataset's metadata, preferring an offline fixture.

    Resolution order: a local ``card_path`` fixture, then (only if
    ``allow_network`` is set) a best-effort network lookup, then an honest
    "absent card" result. Never raises for a missing network.
    """
    if card_path is not None:
        return read_card_file(card_path)
    if allow_network:
        fetched = _fetch_via_hub(dataset_id, revision)
        if fetched is not None:
            return fetched
    return HFDatasetMetadata(
        dataset_id=dataset_id,
        card_present=False,
        revision=_opt_str(revision),
        source="absent",
        note="no local card and network access not enabled",
    )


def _opt_str(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _str_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(v).strip() for v in value if str(v).strip()]
