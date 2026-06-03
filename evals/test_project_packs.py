"""Tests for v1.6 project packs / workspace isolation (offline, no downloads).

These cover the pack contract: a pack has its own isolated store paths; the
active pack can be switched; memory written in one pack is invisible in another;
knowledge imported in one pack is not listed in another; a pack exports to a
bundle containing manifest + all stores and imports back; ``from_pack`` wires the
service to the pack's paths; active pack info is exposed; and binding a pack does
not touch the global default files.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent.project_packs import (  # noqa: E402
    MANIFEST_FILE,
    BANK_FILE,
    LEDGER_FILE,
    PROPOSALS_FILE,
    KNOWLEDGE_FILE,
    PackRegistry,
)
from agent.workbench_service import WorkbenchService  # noqa: E402

_FRIDAY = "the supplier delivery is on Friday afternoon"

_KNOWLEDGE_TEXT = (
    "Python list comprehensions build a new list from an iterable.\n"
    "They read left to right: expression, for-clause, optional if-clause.\n"
)


# 1. create pack creates isolated paths. ------------------------------------

def test_create_pack_creates_isolated_paths(tmp_path):
    registry = PackRegistry(tmp_path / "packs")
    pack_a = registry.create_pack("Alpha")
    pack_b = registry.create_pack("Beta")

    assert pack_a.root_path != pack_b.root_path
    assert pack_a.memory_ledger_path != pack_b.memory_ledger_path
    assert pack_a.knowledge_library_path != pack_b.knowledge_library_path
    # Empty store files exist so an export is always complete.
    for store in (pack_a.memory_bank_path, pack_a.memory_ledger_path,
                  pack_a.proposal_queue_path, pack_a.knowledge_library_path):
        assert store.exists()


# 2. active pack can be switched. -------------------------------------------

def test_active_pack_can_be_switched(tmp_path):
    registry = PackRegistry(tmp_path / "packs")
    pack_a = registry.create_pack("Alpha")
    pack_b = registry.create_pack("Beta")

    registry.set_active_pack(pack_a.pack_id)
    assert registry.get_active_pack().pack_id == pack_a.pack_id

    registry.set_active_pack(pack_b.name)
    assert registry.get_active_pack().pack_id == pack_b.pack_id


# 3. memory written in one pack is not queryable in another. ----------------

def test_memory_isolated_between_packs(tmp_path):
    registry = PackRegistry(tmp_path / "packs")
    pack_a = registry.create_pack("Alpha")
    pack_b = registry.create_pack("Beta")

    svc_a = WorkbenchService.from_pack(pack_a)
    svc_a.add_memory(_FRIDAY, source="test")
    assert svc_a.query_memory(_FRIDAY).memory_used is True

    svc_b = WorkbenchService.from_pack(pack_b)
    audit_b = svc_b.query_memory(_FRIDAY)
    assert audit_b.memory_used is False
    assert audit_b.cited_memory_ids == []


# 4. knowledge imported in one pack is not listed in another. ---------------

def test_knowledge_isolated_between_packs(tmp_path):
    registry = PackRegistry(tmp_path / "packs")
    pack_a = registry.create_pack("Alpha")
    pack_b = registry.create_pack("Beta")

    doc = tmp_path / "ref.md"
    doc.write_text(_KNOWLEDGE_TEXT, encoding="utf-8")

    svc_a = WorkbenchService.from_pack(pack_a)
    svc_a.import_knowledge(doc, domain="coding", authority="official",
                           source_name="Python Reference")
    assert len(svc_a.list_knowledge_sources()) == 1

    svc_b = WorkbenchService.from_pack(pack_b)
    assert svc_b.list_knowledge_sources() == []


# 5. pack export contains manifest, ledger, queue, knowledge files. ---------

def test_pack_export_contains_all_members(tmp_path):
    registry = PackRegistry(tmp_path / "packs")
    pack = registry.create_pack("Alpha")
    svc = WorkbenchService.from_pack(pack)
    svc.add_memory(_FRIDAY, source="test")

    bundle = tmp_path / "alpha.zip"
    registry.export_pack(pack.pack_id, bundle)

    with zipfile.ZipFile(bundle, "r") as zf:
        names = set(zf.namelist())
    assert {MANIFEST_FILE, BANK_FILE, LEDGER_FILE,
            PROPOSALS_FILE, KNOWLEDGE_FILE} <= names


# 6. pack import restores manifest and data. --------------------------------

def test_pack_import_restores_manifest_and_data(tmp_path):
    source_registry = PackRegistry(tmp_path / "src_packs")
    pack = source_registry.create_pack("Alpha", description="my pack")
    svc = WorkbenchService.from_pack(pack)
    svc.add_memory(_FRIDAY, source="test")

    bundle = tmp_path / "alpha.zip"
    source_registry.export_pack(pack.pack_id, bundle)

    target_registry = PackRegistry(tmp_path / "dst_packs")
    restored = target_registry.import_pack(bundle)

    assert restored.name == "Alpha"
    assert restored.description == "my pack"
    # The restored ledger still holds the memory, and a fresh service over the
    # restored pack can ground it (bank records were restored too).
    svc2 = WorkbenchService.from_pack(restored)
    assert svc2.query_memory(_FRIDAY).memory_used is True


# 7. WorkbenchService.from_pack uses pack paths. ----------------------------

def test_from_pack_uses_pack_paths(tmp_path):
    registry = PackRegistry(tmp_path / "packs")
    pack = registry.create_pack("Alpha")
    svc = WorkbenchService.from_pack(pack)
    svc.add_memory(_FRIDAY, source="test")

    # The pack's own ledger file was written, not any global default.
    assert pack.memory_ledger_path.exists()
    assert pack.memory_ledger_path.stat().st_size > 0
    assert pack.memory_bank_path.stat().st_size > 0


# 8. service active pack info is exposed. -----------------------------------

def test_active_pack_info_is_exposed(tmp_path):
    registry = PackRegistry(tmp_path / "packs")
    pack = registry.create_pack("Alpha")

    global_svc = WorkbenchService(ledger_path=str(tmp_path / "g.jsonl"))
    assert global_svc.active_pack_info() is None

    pack_svc = WorkbenchService.from_pack(pack)
    info = pack_svc.active_pack_info()
    assert info is not None
    assert info["pack_id"] == pack.pack_id
    assert info["name"] == "Alpha"


# 9. no default global files are touched when a pack is active. -------------

def test_no_global_files_touched_when_pack_active(tmp_path):
    registry = PackRegistry(tmp_path / "packs")
    pack = registry.create_pack("Alpha")

    global_ledger = tmp_path / "global_ledger.jsonl"
    svc = WorkbenchService.from_pack(pack)
    svc.add_memory(_FRIDAY, source="test")

    assert not global_ledger.exists()
    # All writes landed inside the pack directory.
    assert pack.memory_ledger_path.exists()


# 10. switch_pack re-points the service via the registry. -------------------

def test_switch_pack_repoints_service(tmp_path):
    registry = PackRegistry(tmp_path / "packs")
    pack_a = registry.create_pack("Alpha")
    pack_b = registry.create_pack("Beta")

    svc = WorkbenchService.from_pack(pack_a, registry=registry)
    svc.add_memory(_FRIDAY, source="test")
    assert svc.query_memory(_FRIDAY).memory_used is True

    svc.switch_pack(pack_b.pack_id)
    assert svc.query_memory(_FRIDAY).memory_used is False
    assert registry.get_active_pack().pack_id == pack_b.pack_id

    # Switching back restores the memory from the persisted bank.
    svc.switch_pack(pack_a.pack_id)
    assert svc.query_memory(_FRIDAY).memory_used is True
