<#
.SYNOPSIS
    Build and evaluate the Microsoft 365 consultant pack (v1.9).

.DESCRIPTION
    Builds the M365 consultant knowledge pack (packs\starter\m365_consultant_pack.yaml)
    from local curated Markdown field notes under demos\m365_sources\ only. No web
    access is performed: URL ingestion stays disabled and there is no live
    Microsoft Learn ingestion. The pack is written to an isolated pack root
    (packs\built by default) so the tracked demo packs stay pristine.

    After building, the matching evaluation questions
    (demos\pack_eval_questions\m365_consultant_eval.jsonl) are run and a pass/fail
    report is printed. The script exits non-zero if any evaluation question fails,
    so retrieval and provenance are verified before the pack is trusted for client
    work.

.PARAMETER PackRoot
    Directory to build the pack into (defaults to packs\built).

.PARAMETER SkipEval
    Build the pack but do not run the evaluation questions.

.EXAMPLE
    .\scripts\build_m365_pack.ps1
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
    # the API directly keeps the build robust and never touches the network:
    # allow_url_ingestion stays at its default of False.
    $PyScript = @'
import sys
from pathlib import Path

root = Path(sys.argv[1])
pack_root = Path(sys.argv[2])
run_eval = sys.argv[3] == "1"
sys.path.insert(0, str(root / "src"))

from agent.project_packs import PackRegistry
from agent import pack_builder, pack_evaluator

spec = root / "packs" / "starter" / "m365_consultant_pack.yaml"
eval_path = root / "demos" / "pack_eval_questions" / "m365_consultant_eval.jsonl"

if not spec.exists():
    print(f"M365 pack spec not found: {spec}")
    sys.exit(1)

registry = PackRegistry(pack_root)
plan = pack_builder.PackBuildPlan.from_file(spec)
if not plan.enabled:
    print(f"[skip] {spec.name}: disabled (enabled: false)")
    sys.exit(0)

report = pack_builder.build_pack(plan, registry)
print(f"[built] {report.pack_name} ({report.pack_id}): "
      f"{report.source_count} sources, {report.total_chunks} chunks")
for src in report.sources:
    print(f"    - {src['source_name']} "
          f"[{src['domain']}/{src['authority']}] v{src['version']} "
          f"-> {src['chunks']} chunks")
for skip in report.skipped:
    print(f"    ! skipped {skip['source_name']}: {skip['reason']}")

if not run_eval:
    sys.exit(0)
if not eval_path.exists():
    print(f"eval file not found: {eval_path}")
    sys.exit(1)

service = pack_builder.open_pack_service(registry, report.pack_id)
results = pack_evaluator.evaluate_pack_file(service, eval_path)
passed = sum(1 for r in results if r.passed)
print(f"  eval ({eval_path.name}): {passed}/{len(results)} passed")
failures = 0
for res in results:
    tag = "PASS" if res.passed else "FAIL"
    print(f"    [{tag}] {res.question_id or res.query}: {res.reason}")
    if not res.passed:
        failures += 1

sys.exit(1 if failures else 0)
'@

    $PyScript | & $Python - $RepoRoot $PackRoot $runEval
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
