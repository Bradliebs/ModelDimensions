"""Model round-trips, fingerprints, and import/mutation purity for the v7.3
governed lifecycle action executor."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import agent.lifecycle_action_executor as lae  # noqa: E402
from _lifecycle_executor_helpers import (  # noqa: E402
    FIXED_NOW, approved_action_request, execution_approval, two_pack_manifest,
)

_MODULE_PATH = ROOT / "src" / "agent" / "lifecycle_action_executor.py"

# Functions that must remain pure: they may NEVER call a lifecycle primitive or
# construct a state manager. Only `_apply_state_mutation` and
# `execute_lifecycle_action` are permitted to mutate live state.
_PURE_FUNCTIONS = {
    "validate_lifecycle_execution",
    "build_execution_plan",
    "build_execution_approval",
    "make_operational_follow_up",
    "make_activation_block",
    "verify_post_action",
    "_deactivation_expected_state",
}
_MUTATION_ALLOWED = {"_apply_state_mutation", "execute_lifecycle_action"}

# The executor governs lifecycle actions only; it must never reach into content,
# memory, ingestion, network or scheduling subsystems.
_FORBIDDEN_IMPORT_SUBSTRINGS = (
    "memory_ledger", "source_registry", "memory_proposal", "pack_builder",
    "chunk", "importer", "dataset", "huggingface", "requests", "urllib",
    "httpx", "openai", "anthropic", "apscheduler", "scheduler", "threading",
    "asyncio",
)


def _module_tree() -> ast.Module:
    return ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))


def test_module_imports_are_pure():
    tree = _module_tree()
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    for name in imported:
        for bad in _FORBIDDEN_IMPORT_SUBSTRINGS:
            assert bad not in name, f"forbidden import {name!r} (matched {bad!r})"


def test_pure_functions_never_mutate_state():
    tree = _module_tree()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in _PURE_FUNCTIONS:
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call):
                func = inner.func
                if isinstance(func, ast.Attribute):
                    assert func.attr not in {"deactivate", "rollback"}, (
                        f"{node.name} calls .{func.attr}() — pure functions "
                        "must not invoke lifecycle primitives")
                if isinstance(func, ast.Name):
                    assert func.id != "ActivationStateManager", (
                        f"{node.name} constructs ActivationStateManager")


def test_only_sanctioned_functions_call_primitives():
    """deactivate()/rollback() calls live only in the two sanctioned functions."""
    tree = _module_tree()
    offenders: list[str] = []
    func_stack: list[str] = []

    class _V(ast.NodeVisitor):
        def visit_FunctionDef(self, node):  # noqa: N802
            func_stack.append(node.name)
            self.generic_visit(node)
            func_stack.pop()

        def visit_Call(self, node):  # noqa: N802
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in {
                    "deactivate", "rollback"}:
                current = func_stack[-1] if func_stack else "<module>"
                if current not in _MUTATION_ALLOWED:
                    offenders.append(f"{current}.{func.attr}")
            self.generic_visit(node)

    _V().visit(tree)
    assert not offenders, f"primitive calls outside sanctioned functions: {offenders}"


def test_execution_approval_round_trip():
    ar, _, _ = approved_action_request(mf=two_pack_manifest())
    approval = execution_approval(ar)
    restored = lae.LifecycleExecutionApproval.from_dict(approval.to_dict())
    assert restored == approval


def test_execution_approval_is_deterministic():
    ar, _, _ = approved_action_request(mf=two_pack_manifest())
    a1 = lae.build_execution_approval(
        ar, approved_by="op", approved_role="lifecycle_operator",
        approved_at=FIXED_NOW)
    a2 = lae.build_execution_approval(
        ar, approved_by="op", approved_role="lifecycle_operator",
        approved_at=FIXED_NOW)
    assert a1.execution_approval_id == a2.execution_approval_id


def test_execution_approval_binds_to_action():
    ar, _, _ = approved_action_request(mf=two_pack_manifest())
    approval = execution_approval(ar)
    assert approval.action_request_id == ar.action_request_id
    assert approval.approved_action == ar.requested_action
    assert approval.active_state_hash == ar.active_state_hash
    assert approval.affected_pack_fingerprints == ar.affected_pack_fingerprints


def test_execution_approval_requires_approver():
    ar, _, _ = approved_action_request(mf=two_pack_manifest())
    try:
        lae.build_execution_approval(
            ar, approved_by="  ", approved_role="lifecycle_operator")
    except ValueError:
        return
    raise AssertionError("expected ValueError for blank approver")


def test_layer_version_constant():
    assert lae.EXECUTOR_LAYER_VERSION == "lifecycle-action-executor-v7.3"
