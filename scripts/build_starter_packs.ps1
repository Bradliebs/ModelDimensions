<#
.SYNOPSIS
    Build the local curated knowledge starter packs (v1.8).

.DESCRIPTION
    Builds the starter knowledge packs under packs\starter\ from local demo and
    project documents only. No web access is performed: URL ingestion stays
    disabled, and any .example spec (such as the guarded medical pack) is
    skipped. Built packs are written to an isolated pack root (packs\built by
    default) so the tracked demo packs stay pristine.

    After building, each pack with a matching evaluation file is validated and a
    pass/fail report is printed, so retrieval and provenance are checked before
    the pack is trusted.

.PARAMETER PackRoot
    Directory to build the starter packs into (defaults to packs\built).

.PARAMETER SkipEval
    Build the packs but do not run the evaluation questions.

.EXAMPLE
    .\scripts\build_starter_packs.ps1
#>
[CmdletBinding()]
param(
    [string] $PackRoot,
    [switch] $SkipEval
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Resolve repo root from this script's location, independent of the caller's cwd.
$RepoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $RepoRoot
try {
    # Prefer the project venv interpreter; fall back to 'python' on PATH.
    $VenvPython = Join-Path $RepoRoot '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $VenvPython) {
        $Python = $VenvPython
    }
    else {
        $Python = 'python'
        Write-Warning "Project venv not found at $VenvPython; using 'python' on PATH."
    }

    if (-not $PackRoot) { $PackRoot = Join-Path $RepoRoot 'packs\built' }

    $runEval = '1'
    if ($SkipEval) { $runEval = '0' }

    # Drive the build through the pack_builder/pack_evaluator module API. Using
    # the API directly (rather than piping REPL commands) keeps the build robust
    # and never touches the network: allow_url_ingestion stays at its default.
    $PyScript = @'
import sys
from pathlib import Path

root = Path(sys.argv[1])
pack_root = Path(sys.argv[2])
run_eval = sys.argv[3] == "1"
sys.path.insert(0, str(root / "src"))

from agent.project_packs import PackRegistry
from agent import pack_builder, pack_evaluator

# Starter specs build only local sources; .example specs are skipped here.
starter_dir = root / "packs" / "starter"
eval_dir = root / "demos" / "pack_eval_questions"
# Explicit spec -> eval mapping keeps validation transparent.
eval_map = {
    "coding_reference_pack.yaml": "coding_reference_eval.jsonl",
    "concept_cells_project_pack.yaml": "concept_cells_eval.jsonl",
}

registry = PackRegistry(pack_root)
specs = sorted(p for p in starter_dir.glob("*.yaml") if p.suffix == ".yaml")
if not specs:
    print(f"no starter specs found in {starter_dir}")
    sys.exit(1)

failures = 0
for spec in specs:
    plan = pack_builder.PackBuildPlan.from_file(spec)
    if not plan.enabled:
        print(f"[skip] {spec.name}: disabled (enabled: false)")
        continue
    report = pack_builder.build_pack(plan, registry)
    print(f"[built] {report.pack_name} ({report.pack_id}): "
          f"{report.source_count} sources, {report.total_chunks} chunks")
    for src in report.sources:
        print(f"    - {src['source_name']} "
              f"[{src['domain']}/{src['authority']}] -> {src['chunks']} chunks")
    for skip in report.skipped:
        print(f"    ! skipped {skip['source_name']}: {skip['reason']}")

    if not run_eval:
        continue
    eval_name = eval_map.get(spec.name)
    if not eval_name:
        continue
    eval_path = eval_dir / eval_name
    if not eval_path.exists():
        continue
    service = pack_builder.open_pack_service(registry, report.pack_id)
    results = pack_evaluator.evaluate_pack_file(service, eval_path)
    passed = sum(1 for r in results if r.passed)
    print(f"  eval ({eval_name}): {passed}/{len(results)} passed")
    for res in results:
        if not res.passed:
            failures += 1
            print(f"    [FAIL] {res.question_id or res.query}: {res.reason}")

sys.exit(1 if failures else 0)
'@

    $PyScript | & $Python - $RepoRoot $PackRoot $runEval
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
