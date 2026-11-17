param([string]$Python311, [string]$EnvironmentPath='.venv')
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'windows-runtime.ps1')
$runtime=Initialize-ResearchRuntime -Root $PSScriptRoot -Python311 $Python311 -EnvironmentPath $EnvironmentPath
& $runtime.Python -m pytest -q
if($LASTEXITCODE -ne 0){throw 'Tests failed'}
& $runtime.Python -m ruff check src tests
if($LASTEXITCODE -ne 0){throw 'Lint failed'}
& $runtime.Python -m pip check
if($LASTEXITCODE -ne 0){throw 'Dependency check failed'}
& $runtime.Python -m model_assurance.cli run --config configs/experiment.yaml
if($LASTEXITCODE -ne 0){throw 'Baseline failed'}
