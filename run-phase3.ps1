param([string]$Python311, [string]$EnvironmentPath='.venv', [switch]$InstallDependencies, [switch]$Offline)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'windows-runtime.ps1')
$runtime=Initialize-ResearchRuntime -Root $PSScriptRoot -Python311 $Python311 -EnvironmentPath $EnvironmentPath -InstallDependencies:$InstallDependencies
$stamp=Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$output="results/phase3-$stamp"
$data=if(Test-Path evidence/datasets){'evidence/datasets'}else{'results/external-cache'}
$arguments=@('-m','model_assurance.external_validation','--config','configs/phase3.yaml','--cache',$data,'--output',$output)
if($Offline){$arguments+='--offline'}
& $runtime.Python @arguments
if($LASTEXITCODE -ne 0){throw 'Phase 3 failed'}
& $runtime.Python -m model_assurance.verify_external --output $output --data-dir $data --record "reports/verification-$stamp.json"
if($LASTEXITCODE -ne 0){throw 'Semantic verification failed'}
Write-Host "Exploratory results: $output; detached verification record: reports/verification-$stamp.json"
