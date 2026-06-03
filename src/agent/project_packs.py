"""Project packs: isolated workspaces for the Concept Memory Workbench (v1.6).

A *pack* is a self-contained directory holding one project's stores:

    <root>/<pack_id>/
        manifest.json     # pack metadata + runtime settings
        bank.jsonl        # persisted concept-cell records (vectors + ids)
        ledger.jsonl      # human-facing memory ledger
        proposals.jsonl   # ingestion proposal queue
        knowledge.jsonl   # imported knowledge library

A :class:`PackRegistry` owns a root directory of packs plus a tiny
``registry.json`` that records which pack is active. Packs are fully isolated:
each one has its own files, so a memory or knowledge import in one pack is never
visible from another. A pack can be exported to a single ``.zip`` bundle and
imported back, restoring its manifest and all four stores.

This module adds *no* geometry and touches none of the frozen v1.0 path. It only
manages file locations and metadata; the :class:`~agent.workbench_service.WorkbenchService`
binds its existing stores to a pack's paths.
"""
from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Stable, per-pack file names. Kept as module constants so a pack directory is
# always self-describing and export/import is robust to manifest drift.
MANIFEST_FILE = "manifest.json"
BANK_FILE = "bank.jsonl"
LEDGER_FILE = "ledger.jsonl"
PROPOSALS_FILE = "proposals.jsonl"
KNOWLEDGE_FILE = "knowledge.jsonl"

# The four stores that make up a pack's exportable state (manifest + data).
_BUNDLE_FILES = (MANIFEST_FILE, BANK_FILE, LEDGER_FILE, PROPOSALS_FILE,
                 KNOWLEDGE_FILE)

_REGISTRY_FILE = "registry.json"
_DEFAULT_BACKEND = "deterministic"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(name: str) -> str:
    """Turn a display name into a filesystem-safe pack id stem."""
    slug = re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower())
    slug = slug.strip("_")
    return slug or "pack"


@dataclass
class PackManifest:
    """Metadata and runtime settings for one project pack.

    Persisted verbatim to ``manifest.json``. ``settings`` is an open dict for
    per-pack runtime configuration (e.g. seed files for the demo packs); unknown
    keys are preserved on round-trip so the format can grow without breaking
    older packs.
    """

    pack_id: str
    name: str
    description: str = ""
    created_at: str = field(default_factory=_utc_now)
    default_knowledge_backend: str = _DEFAULT_BACKEND
    default_domain: Optional[str] = None
    settings: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "pack_id": self.pack_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "default_knowledge_backend": self.default_knowledge_backend,
            "default_domain": self.default_domain,
            "settings": dict(self.settings),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PackManifest":
        return cls(
            pack_id=data["pack_id"],
            name=data.get("name", data["pack_id"]),
            description=data.get("description", ""),
            created_at=data.get("created_at", _utc_now()),
            default_knowledge_backend=data.get(
                "default_knowledge_backend", _DEFAULT_BACKEND),
            default_domain=data.get("default_domain"),
            settings=dict(data.get("settings", {})),
        )


@dataclass
class ProjectPack:
    """One pack: its manifest plus the directory that holds its stores.

    The resolved store paths are properties off ``root_path`` so the flat field
    list the workbench needs (ledger, queue, knowledge, bank) is available
    without duplicating state.
    """

    manifest: PackManifest
    root_path: Path

    # -- convenience accessors for the flat metadata fields --

    @property
    def pack_id(self) -> str:
        return self.manifest.pack_id

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def description(self) -> str:
        return self.manifest.description

    @property
    def created_at(self) -> str:
        return self.manifest.created_at

    @property
    def default_knowledge_backend(self) -> str:
        return self.manifest.default_knowledge_backend

    @property
    def default_domain(self) -> Optional[str]:
        return self.manifest.default_domain

    @property
    def settings(self) -> Dict[str, object]:
        return self.manifest.settings

    # -- resolved store paths --

    @property
    def manifest_path(self) -> Path:
        return self.root_path / MANIFEST_FILE

    @property
    def memory_bank_path(self) -> Path:
        return self.root_path / BANK_FILE

    @property
    def memory_ledger_path(self) -> Path:
        return self.root_path / LEDGER_FILE

    @property
    def proposal_queue_path(self) -> Path:
        return self.root_path / PROPOSALS_FILE

    @property
    def knowledge_library_path(self) -> Path:
        return self.root_path / KNOWLEDGE_FILE

    def resolve_setting_path(self, key: str) -> Optional[Path]:
        """Resolve a settings value naming a file relative to the pack dir."""
        value = self.settings.get(key)
        if not value:
            return None
        candidate = (self.root_path / str(value)).resolve()
        return candidate if candidate.exists() else None

    def to_info_dict(self) -> dict:
        """A display/audit summary of this pack (paths + line counts)."""
        return {
            "pack_id": self.pack_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "default_knowledge_backend": self.default_knowledge_backend,
            "default_domain": self.default_domain,
            "root_path": str(self.root_path),
            "memory_ledger_path": str(self.memory_ledger_path),
            "proposal_queue_path": str(self.proposal_queue_path),
            "knowledge_library_path": str(self.knowledge_library_path),
            "memory_bank_path": str(self.memory_bank_path),
            "memory_entries": _count_lines(self.memory_ledger_path),
            "proposal_entries": _count_lines(self.proposal_queue_path),
            "knowledge_records": _count_lines(self.knowledge_library_path),
        }


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


class PackRegistry:
    """Owns a root directory of packs and tracks which one is active."""

    def __init__(self, root_path: str | Path):
        self.root_path = Path(root_path)
        self.root_path.mkdir(parents=True, exist_ok=True)

    # -- internal state file --

    @property
    def _state_path(self) -> Path:
        return self.root_path / _REGISTRY_FILE

    def _read_state(self) -> dict:
        if not self._state_path.exists():
            return {}
        try:
            return json.loads(self._state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _write_state(self, state: dict) -> None:
        self._state_path.write_text(
            json.dumps(state, indent=2), encoding="utf-8")

    # -- discovery --

    def list_packs(self) -> List[ProjectPack]:
        """Return every pack under the root, sorted by creation time."""
        packs: List[ProjectPack] = []
        for child in sorted(self.root_path.iterdir()):
            if not child.is_dir():
                continue
            manifest_path = child / MANIFEST_FILE
            if not manifest_path.exists():
                continue
            pack = self._load_pack(child)
            if pack is not None:
                packs.append(pack)
        packs.sort(key=lambda p: p.created_at)
        return packs

    def _load_pack(self, root: Path) -> Optional[ProjectPack]:
        manifest_path = root / MANIFEST_FILE
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return ProjectPack(PackManifest.from_dict(data), root)

    def get_pack(self, pack_id_or_name: str) -> Optional[ProjectPack]:
        """Find a pack by exact id first, then by exact display name."""
        direct = self.root_path / pack_id_or_name
        if (direct / MANIFEST_FILE).exists():
            return self._load_pack(direct)
        for pack in self.list_packs():
            if pack.pack_id == pack_id_or_name or pack.name == pack_id_or_name:
                return pack
        return None

    # -- create --

    def _unique_pack_id(self, name: str) -> str:
        base = _slugify(name)
        if not (self.root_path / base).exists():
            return base
        suffix = 2
        while (self.root_path / f"{base}_{suffix}").exists():
            suffix += 1
        return f"{base}_{suffix}"

    def create_pack(self, name: str,
                    description: str = "",
                    default_knowledge_backend: str = _DEFAULT_BACKEND,
                    default_domain: Optional[str] = None,
                    settings: Optional[Dict[str, object]] = None) -> ProjectPack:
        """Create a new, empty pack directory and its four (empty) store files."""
        if not (name or "").strip():
            raise ValueError("pack name must be non-empty")
        pack_id = self._unique_pack_id(name)
        root = self.root_path / pack_id
        root.mkdir(parents=True, exist_ok=False)
        manifest = PackManifest(
            pack_id=pack_id,
            name=name.strip(),
            description=description or "",
            default_knowledge_backend=default_knowledge_backend,
            default_domain=default_domain,
            settings=dict(settings or {}),
        )
        pack = ProjectPack(manifest, root)
        self._save_manifest(pack)
        # Create empty store files so an export always contains all members.
        for store in (pack.memory_bank_path, pack.memory_ledger_path,
                      pack.proposal_queue_path, pack.knowledge_library_path):
            if not store.exists():
                store.write_text("", encoding="utf-8")
        return pack

    def _save_manifest(self, pack: ProjectPack) -> None:
        pack.manifest_path.write_text(
            json.dumps(pack.manifest.to_dict(), indent=2), encoding="utf-8")

    # -- active pack --

    def set_active_pack(self, pack_id_or_name: str) -> ProjectPack:
        pack = self.get_pack(pack_id_or_name)
        if pack is None:
            raise KeyError(f"no such pack: {pack_id_or_name!r}")
        state = self._read_state()
        state["active_pack_id"] = pack.pack_id
        self._write_state(state)
        return pack

    def get_active_pack(self) -> Optional[ProjectPack]:
        active_id = self._read_state().get("active_pack_id")
        if not active_id:
            return None
        return self.get_pack(active_id)

    def pack_info(self, pack_id_or_name: str) -> Optional[dict]:
        pack = self.get_pack(pack_id_or_name)
        if pack is None:
            return None
        info = pack.to_info_dict()
        active = self.get_active_pack()
        info["active"] = active is not None and active.pack_id == pack.pack_id
        return info

    # -- export / import --

    def export_pack(self, pack_id_or_name: str,
                    output_path: str | Path) -> Path:
        """Zip a pack's manifest and four stores into ``output_path``."""
        pack = self.get_pack(pack_id_or_name)
        if pack is None:
            raise KeyError(f"no such pack: {pack_id_or_name!r}")
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as bundle:
            for member in _BUNDLE_FILES:
                source = pack.root_path / member
                # Ensure every member exists so the bundle is complete.
                data = source.read_text(encoding="utf-8") if source.exists() else ""
                bundle.writestr(member, data)
        return out

    def import_pack(self, bundle_path: str | Path) -> ProjectPack:
        """Restore a pack from a ``.zip`` bundle, registering it under its id."""
        src = Path(bundle_path)
        with zipfile.ZipFile(src, "r") as bundle:
            try:
                manifest_raw = bundle.read(MANIFEST_FILE).decode("utf-8")
            except KeyError as exc:  # pragma: no cover - defensive
                raise ValueError("bundle has no manifest.json") from exc
            manifest = PackManifest.from_dict(json.loads(manifest_raw))
            root = self.root_path / manifest.pack_id
            root.mkdir(parents=True, exist_ok=True)
            for member in _BUNDLE_FILES:
                try:
                    data = bundle.read(member)
                except KeyError:
                    data = b""
                (root / member).write_bytes(data)
        return ProjectPack(manifest, root)
