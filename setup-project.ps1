param([string]$Python311, [string]$EnvironmentPath='.venv')
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'windows-runtime.ps1')
$runtime=Initialize-ResearchRuntime -Root $PSScriptRoot -Python311 $Python311 -EnvironmentPath $EnvironmentPath -InstallDependencies
& $runtime.Python -m pip check
if($LASTEXITCODE -ne 0){throw 'Dependency check failed'}
Write-Host 'Environment ready. No experiment or existing result was overwritten.'
