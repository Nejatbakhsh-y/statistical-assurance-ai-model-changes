param([string]$Python311, [string]$EnvironmentPath='.venv', [switch]$InstallDependencies)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'windows-runtime.ps1')
$runtime=Initialize-ResearchRuntime -Root $PSScriptRoot -Python311 $Python311 -EnvironmentPath $EnvironmentPath -InstallDependencies:$InstallDependencies
$output='results/phase2-'+(Get-Date -Format 'yyyyMMdd-HHmmss-fff')
& $runtime.Python -m model_assurance.robustness --config configs/phase2.yaml --output $output
if($LASTEXITCODE -ne 0){throw 'Phase 2 failed'}
Write-Host $output
