<#
.SYNOPSIS
    Import a tiny, governed Hugging Face coding sample into a pack (v1.9).

.DESCRIPTION
    Demonstrates the licence-aware HF dataset importer using the bundled OFFLINE
    fixtures under demos\hf_fixtures only. This script is disabled-safe: it never
    reaches the network. Metadata is read from a local dataset card and rows are
    read from a local JSONL fixture (allow_network stays False), so no dataset is
    ever downloaded.

    By default it imports a handful of coding rows in EVAL mode into an isolated
    pack root (packs\built). It prints the licence/source decision and any
    warnings so the governance checks are visible before any data is trusted.

    Free does not mean trusted: an unknown licence is blocked for knowledge mode,
    medical/legal domains require a known authority, and the sample size is
    capped. This demo stays well inside those rails.

.PARAMETER PackRoot
    Directory to build the pack into (defaults to packs\built).

.PARAMETER PackName
    Name of the pack to import into (created if absent; defaults to hf-demo).

.PARAMETER Mode
    Import mode: 'eval' (default) or 'knowledge'.

.EXAMPLE
    .\scripts\import_hf_sample.ps1

.EXAMPLE
    .\scripts\import_hf_sample.ps1 -Mode knowledge
#>
[CmdletBinding()]
param(
    [string] $PackRoot,
    [string] $PackName = 'hf-demo',
    [ValidateSet('eval', 'knowledge')]
    [string] $Mode = 'eval'
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

    Write-Host "HF sample import is OFFLINE and disabled-safe: no network access." `
        -ForegroundColor Cyan
    Write-Host "Reminder: free does not mean trusted. Verify licence and source." `
        -ForegroundColor Yellow

    # Drive the import through the importer module API directly. allow_network is
    # never set, and the rows come from a local fixture, so nothing downloads.
    $PyScript = @'
import sys
from pathlib import Path

root = Path(sys.argv[1])
pack_root = Path(sys.argv[2])
pack_name = sys.argv[3]
mode = sys.argv[4]
sys.path.insert(0, str(root / "src"))

from agent.project_packs import PackRegistry
from agent.hf_dataset_importer import HFDatasetSpec, import_hf_dataset_to_pack

fixtures = root / "demos" / "hf_fixtures"
card = fixtures / "dataset_card_with_license.json"
rows = fixtures / "small_coding_dataset.jsonl"
if not card.exists() or not rows.exists():
    print(f"missing offline fixtures under {fixtures}")
    sys.exit(1)

registry = PackRegistry(pack_root)
pack = registry.get_pack(pack_name) or registry.create_pack(
    pack_name, description="HF importer demo")

spec = HFDatasetSpec(
    dataset_id="demo/small-coding",
    split="train",
    text_fields=["text"],
    sample_size=5,
    domain="coding",
    authority="reputable",
    mode=mode,
    card_path=str(card),
    local_fixture=str(rows),
    allow_network=False,
)

report = import_hf_dataset_to_pack(spec, pack)
print(f"[hf-import] {report.dataset_id} -> pack {pack.name} ({mode})")
print(f"    licence      = {report.license or '-'} "
      f"(known: {str(report.license_known).lower()})")
print(f"    accepted     = {str(report.accepted).lower()}")
if not report.accepted:
    print(f"    blocked      = {report.rejected_reason}")
else:
    print(f"    imported     = {report.imported_count} rows")
    if report.knowledge_source_id:
        print(f"    knowledge    = {report.knowledge_source_id}")
    if report.eval_output_path:
        print(f"    eval_samples = {report.eval_output_path}")
for warn in report.warnings:
    print(f"    ! warning    = {warn}")

sys.exit(0 if report.accepted else 1)
'@

    $PyScript | & $Python - $RepoRoot $PackRoot $PackName $Mode
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
