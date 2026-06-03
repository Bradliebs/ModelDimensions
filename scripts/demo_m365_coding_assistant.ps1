<#
.SYNOPSIS
    Build, evaluate, and demo the M365 + coding assistant pack (v2.0).

.DESCRIPTION
    Builds the first real operational assistant pack
    (packs\starter\m365_coding_assistant.yaml) from local curated Markdown field
    notes under demos\m365_coding_sources\ only. No web access is performed: URL
    ingestion stays disabled and there is no live Microsoft Learn ingestion. The
    pack is written to an isolated pack root (packs\built by default) so the
    tracked demo packs stay pristine.

    After building, the matching evaluation questions
    (demos\pack_eval_questions\m365_coding_assistant_eval.jsonl) are run and a
    pass/fail report is printed. Then five example questions are answered through
    the combined query path, printing the memory_used / knowledge_used /
    model_prior_used routing flags for each so the separation of project memory,
    imported knowledge, and ungrounded model prior is visible.

    The script exits non-zero if any evaluation question fails, so retrieval and
    provenance are verified before the pack is trusted.

.PARAMETER PackRoot
    Directory to build the pack into (defaults to packs\built).

.PARAMETER SkipEval
    Build the pack and run the demo but do not gate on the evaluation questions.

.EXAMPLE
    .\scripts\demo_m365_coding_assistant.ps1
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

    # Drive the whole demo through the pack_builder / pack_evaluator /
    # WorkbenchService module API. Using the API directly keeps it robust and
    # never touches the network: allow_url_ingestion stays at its default False.
    $PyScript = @'
import sys
from pathlib import Path

root = Path(sys.argv[1])
pack_root = Path(sys.argv[2])
run_eval = sys.argv[3] == "1"
sys.path.insert(0, str(root / "src"))

from agent.project_packs import PackRegistry
from agent import pack_builder, pack_evaluator

spec = root / "packs" / "starter" / "m365_coding_assistant.yaml"
eval_path = (root / "demos" / "pack_eval_questions"
             / "m365_coding_assistant_eval.jsonl")

if not spec.exists():
    print(f"assistant pack spec not found: {spec}")
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

service = pack_builder.open_pack_service(registry, report.pack_id)

# --- Eval gate ----------------------------------------------------------
failures = 0
if run_eval:
    if not eval_path.exists():
        print(f"eval file not found: {eval_path}")
        sys.exit(1)
    results = pack_evaluator.evaluate_pack_file(service, eval_path)
    passed = sum(1 for r in results if r.passed)
    print(f"\n  eval ({eval_path.name}): {passed}/{len(results)} passed")
    for res in results:
        tag = "PASS" if res.passed else "FAIL"
        print(f"    [{tag}] {res.question_id or res.query}: {res.reason}")
        if not res.passed:
            failures += 1

# --- Routing-flag demonstration ----------------------------------------
# Pick a couple of real source chunks so knowledge retrieval lands on the
# intended source, plus decision-recall queries that route to project memory
# only. With an empty ledger those fall back to the ungrounded model prior, so
# all three flags are exercised.
chunks = service.knowledge.list_chunks(active_only=True)


def chunk_for(source_name):
    for c in chunks:
        if c.source_name == source_name and c.source_section == "Key facts":
            return c.chunk_text
    return source_name


examples = [
    ("Pydantic v2: how is data validated?", chunk_for("Pydantic v2 Notes")),
    ("Purview: how do sensitivity labels protect content?",
     chunk_for("Purview Sensitivity Labels")),
    ("PowerShell: how does a launcher propagate an exit code?",
     chunk_for("PowerShell Launcher Notes")),
    ("What did we decide about the production database?",
     "What did we decide about the production database?"),
    ("What did we agree on for the release plan?",
     "What did we agree on for the release plan?"),
]

print("\n  example questions (memory_used / knowledge_used / model_prior_used):")
for label, query in examples:
    audit = service.query_all(query)
    print(f"    - {label}")
    print(f"        memory_used={audit.memory_used} "
          f"knowledge_used={audit.knowledge_used} "
          f"model_prior_used={audit.model_prior_used} "
          f"(route={audit.route})")

sys.exit(1 if failures else 0)
'@

    $PyScript | & $Python - $RepoRoot $PackRoot $runEval
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
