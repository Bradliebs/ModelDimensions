<#
.SYNOPSIS
    Launch the Concept Memory Review Console (v1.7) locally.

.DESCRIPTION
    Starts the offline operator console over the frozen v1.0 memory. If
    Streamlit is installed in the project venv it launches the web UI; otherwise
    it falls back to the interactive CLI. Both run fully offline and enforce the
    same approval gates and grounding policy as the workbench — the console only
    surfaces them, it never bypasses them.

.PARAMETER Pack
    Operate the named project pack (isolated workspace). Forces the CLI, because
    Streamlit does not forward script arguments.

.PARAMETER PackRoot
    Directory holding project packs (defaults to demos\packs).

.EXAMPLE
    .\scripts\run_review_console.ps1

.EXAMPLE
    .\scripts\run_review_console.ps1 -Pack concept-cells
#>
[CmdletBinding()]
param(
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

    $AppPath = Join-Path $RepoRoot 'app\review_console.py'

    # Detect Streamlit without failing if it is absent.
    $hasStreamlit = $false
    & $Python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('streamlit') else 1)"
    if ($LASTEXITCODE -eq 0) { $hasStreamlit = $true }

    $cliArgs = @()
    if ($PackRoot) { $cliArgs += @('--pack-root', $PackRoot) }
    if ($Pack) { $cliArgs += @('--pack', $Pack) }

    # A specific pack means CLI mode: Streamlit ignores script args, so the pack
    # would be silently dropped under the web UI.
    if ($hasStreamlit -and -not $Pack) {
        Write-Host 'Streamlit found: launching the web review console...' -ForegroundColor Green
        & $Python -m streamlit run $AppPath
    }
    else {
        if (-not $hasStreamlit) {
            Write-Host 'Streamlit is not installed: launching the CLI review console.' -ForegroundColor Yellow
            Write-Host 'To use the web UI instead, install it into the venv:' -ForegroundColor Yellow
            Write-Host '    .\.venv\Scripts\python.exe -m pip install streamlit' -ForegroundColor Yellow
        }
        else {
            Write-Host 'Pack requested: launching the CLI review console...' -ForegroundColor Yellow
        }
        & $Python $AppPath @cliArgs
    }
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
