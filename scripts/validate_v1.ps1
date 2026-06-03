<#
.SYNOPSIS
    One-command reproduction of the Concept Cells v1.0-research-baseline.

.DESCRIPTION
    Runs the full validation chain in a fresh shell and verifies the expected
    result artifacts were produced. Intended as the "fresh reproduction" proof
    for the v1.0 freeze: a single command that confirms the system is intact.

    Steps:
        1. pytest                              (full suite)
        2. experiments.exp01_smoke             (mechanism smoke test)
        3. experiments.exp10_candidate_verifier
        4. experiments.exp11_failure_taxonomy  (15/15 flip taxonomy)
        5. experiments.exp12_paraphrase_recovery
        6. scripts/demo_v1.py                  (end-to-end demo)

    Then checks that these exist:
        results/exp10_summary.json
        results/exp11_summary.json
        results/exp12_summary.json
        docs/v1_release_report.md

    Exits 0 only if every step succeeds and every artifact is present.

.NOTES
    Optional NLI (Exp 12) stays disabled by default; this script does not set
    EXP12_ENABLE_NLI, so the offline default path is what gets validated.

.EXAMPLE
    .\scripts\validate_v1.ps1
#>
[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Resolve repo root from this script's location, independent of the caller's cwd.
$RepoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $RepoRoot

# Prefer the project venv interpreter; fall back to whatever 'python' resolves to.
$VenvPython = Join-Path $RepoRoot '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $VenvPython) {
    $Python = $VenvPython
}
else {
    $Python = 'python'
    Write-Warning "Project venv not found at $VenvPython; using 'python' on PATH."
}

$failures = [System.Collections.Generic.List[string]]::new()

function Invoke-Step {
    param(
        [Parameter(Mandatory)] [string] $Name,
        [Parameter(Mandatory)] [string[]] $Arguments
    )

    Write-Host ''
    Write-Host ('=' * 70) -ForegroundColor Cyan
    Write-Host "STEP: $Name" -ForegroundColor Cyan
    Write-Host ('=' * 70) -ForegroundColor Cyan

    & $Python @Arguments
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        $script:failures.Add("$Name (exit $code)")
        Write-Host "FAILED: $Name (exit $code)" -ForegroundColor Red
    }
    else {
        Write-Host "OK: $Name" -ForegroundColor Green
    }
}

try {
    Invoke-Step -Name 'pytest (full suite)' -Arguments @('-m', 'pytest', '-q')
    Invoke-Step -Name 'exp01 smoke' -Arguments @('-m', 'experiments.exp01_smoke')
    Invoke-Step -Name 'exp10 candidate verifier' -Arguments @('-m', 'experiments.exp10_candidate_verifier')
    Invoke-Step -Name 'exp11 failure taxonomy' -Arguments @('-m', 'experiments.exp11_failure_taxonomy')
    Invoke-Step -Name 'exp12 paraphrase recovery' -Arguments @('-m', 'experiments.exp12_paraphrase_recovery')
    Invoke-Step -Name 'demo_v1' -Arguments @('scripts/demo_v1.py')

    Write-Host ''
    Write-Host ('=' * 70) -ForegroundColor Cyan
    Write-Host 'ARTIFACT CHECK' -ForegroundColor Cyan
    Write-Host ('=' * 70) -ForegroundColor Cyan

    $artifacts = @(
        'results/exp10_summary.json',
        'results/exp11_summary.json',
        'results/exp12_summary.json',
        'docs/v1_release_report.md'
    )
    foreach ($artifact in $artifacts) {
        $path = Join-Path $RepoRoot $artifact
        if (Test-Path -LiteralPath $path) {
            Write-Host "OK: $artifact" -ForegroundColor Green
        }
        else {
            $failures.Add("missing artifact: $artifact")
            Write-Host "MISSING: $artifact" -ForegroundColor Red
        }
    }

    Write-Host ''
    Write-Host ('=' * 70) -ForegroundColor Cyan
    if ($failures.Count -eq 0) {
        Write-Host 'v1.0 REPRODUCTION: PASS' -ForegroundColor Green
        Write-Host ('=' * 70) -ForegroundColor Cyan
        exit 0
    }
    else {
        Write-Host 'v1.0 REPRODUCTION: FAIL' -ForegroundColor Red
        foreach ($failure in $failures) {
            Write-Host "  - $failure" -ForegroundColor Red
        }
        Write-Host ('=' * 70) -ForegroundColor Cyan
        exit 1
    }
}
finally {
    Pop-Location
}
