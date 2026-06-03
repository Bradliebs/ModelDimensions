<#
.SYNOPSIS
    Launch the Concept Memory Workbench (v1.1) locally.

.DESCRIPTION
    Starts the offline workbench over the frozen v1.0 memory. If Streamlit is
    installed in the project venv it launches the web UI; otherwise it falls
    back to the interactive CLI. Both run fully offline.

    The workbench seeds the project memories by default. Pass -NoSeed to start
    empty, or -Demo to run the Friday -> Monday near-miss example and exit.

.PARAMETER NoSeed
    Start with an empty workbench instead of seeding the project memories.

.PARAMETER Demo
    Run the Friday -> Monday example through the service and exit (CLI only).

.PARAMETER Pack
    Open the named project pack (isolated workspace). Forces the CLI, because
    Streamlit does not forward script arguments.

.PARAMETER PackRoot
    Directory holding project packs (defaults to demos\packs).

.EXAMPLE
    .\scripts\run_workbench.ps1

.EXAMPLE
    .\scripts\run_workbench.ps1 -Demo

.EXAMPLE
    .\scripts\run_workbench.ps1 -Pack concept-cells
#>
[CmdletBinding()]
param(
    [switch] $NoSeed,
    [switch] $Demo,
    [string] $Pack,
    [string] $PackRoot
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

    $AppPath = Join-Path $RepoRoot 'app\workbench.py'

    # Detect Streamlit without failing if it is absent.
    $hasStreamlit = $false
    & $Python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('streamlit') else 1)"
    if ($LASTEXITCODE -eq 0) { $hasStreamlit = $true }

    if ($Demo) {
        # The demo is a CLI-only walkthrough.
        & $Python $AppPath '--demo'
        exit $LASTEXITCODE
    }

    $cliArgs = @()
    if ($NoSeed) { $cliArgs += '--no-seed' }
    if ($PackRoot) { $cliArgs += @('--pack-root', $PackRoot) }
    if ($Pack) { $cliArgs += @('--pack', $Pack) }

    # A specific pack means CLI mode: Streamlit ignores script args, so the pack
    # would be silently dropped under the web UI.
    if ($hasStreamlit -and -not $Pack) {
        Write-Host 'Streamlit found: launching the web workbench...' -ForegroundColor Green
        & $Python -m streamlit run $AppPath
    }
    else {
        Write-Host 'Streamlit not installed or pack requested: launching the CLI workbench...' -ForegroundColor Yellow
        & $Python $AppPath @cliArgs
    }
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
