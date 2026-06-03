# PowerShell Launcher Notes

Source: assistant local field notes
Version: local-notes-v1
Domain: coding
Authority: reputable
Staleness: review_required (verify against the PowerShell docs before relying on it)

## Key facts

- `$LASTEXITCODE` holds the exit code of the last native executable, and
  `exit $LASTEXITCODE` propagates a child process failure to the caller.
- A launcher script should set `Set-StrictMode -Version Latest` and
  `$ErrorActionPreference = 'Stop'` so unset variables and failed commands halt
  the run instead of continuing silently.
- `$PSScriptRoot` is the directory of the running script; deriving the repo root
  with `Split-Path -Parent $PSScriptRoot` keeps paths stable regardless of the
  caller's working directory.
- `Push-Location` / `Pop-Location` in a try/finally block run a script from a
  known directory and always restore the caller's location.
- `& $Python script.py` invokes an executable by path; the call operator `&` is
  required when the command is stored in a variable.

## Decisions and assumptions

- Decision: detect a virtual-environment Python at `.venv\Scripts\python.exe`
  and fall back to `python` on PATH when it is absent.
- Assumption: scripts target PowerShell 5.1 and PowerShell 7+, avoiding syntax
  exclusive to either.
- Decision: pass the repo root and other paths as explicit arguments to embedded
  Python rather than relying on the inherited working directory.

## Known limitations

- `$LASTEXITCODE` reflects native executables only; a failing PowerShell cmdlet
  sets `$?` instead, so both must be checked when mixing the two.
- StrictMode turns referencing an undefined variable into a terminating error,
  which can surprise scripts that relied on empty defaults.
- A here-string used to embed Python must use the single-quoted `@'...'@` form to
  stop PowerShell from expanding `$` variables inside the Python code.

## Examples

- Example: `$Python = if (Test-Path $venv) { $venv } else { 'python' }` selects
  the interpreter with a clear fallback.
- Example: `try { Push-Location $RepoRoot; & $Python - $RepoRoot } finally {
  Pop-Location }` runs from the root and always restores location.

## Caution / near-miss

- Near-miss: a launcher embedded Python with a double-quoted here-string, so
  PowerShell expanded `$service` inside the Python source and produced a syntax
  error at runtime. Switching to the single-quoted `@'...'@` form fixed it.
