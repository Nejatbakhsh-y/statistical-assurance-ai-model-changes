# Shared Windows entry point. No registry, PATH, tool installation or destructive repair.
function Initialize-ResearchRuntime {
  param([string]$Root, [string]$Python311, [string]$EnvironmentPath = '.venv', [switch]$InstallDependencies)
  $ErrorActionPreference = 'Stop'
  $candidates = @()
  if ($Python311) { $candidates = @($Python311) }
  else {
    $candidates += (Join-Path $env:LOCALAPPDATA 'Programs/Python/Python311/python.exe')
    if (Get-Command py -ErrorAction SilentlyContinue) {
      $launcherPath = & py -3.11 -c 'import sys; print(sys.executable)' 2>$null
      if ($LASTEXITCODE -eq 0) { $candidates += $launcherPath }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) { $candidates += (Get-Command python).Source }
  }
  $base = $null
  foreach ($candidate in $candidates) {
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { continue }
    & $candidate -B -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3,11) else 1)'
    if ($LASTEXITCODE -eq 0) { $base = $candidate; break }
  }
  if (-not $base) { throw 'Supply -Python311 with an actual Python 3.11 executable; no interpreter was installed or replaced.' }
  if (-not (Test-Path R:\)) {
    & subst R: $Root
    if ($LASTEXITCODE -ne 0) { throw 'Cannot create temporary R: mapping' }
  }
  & $base -B -c 'import os,sys; sys.exit(0 if os.path.samefile(sys.argv[1],sys.argv[2]) else 1)' $Root 'R:\'
  if ($LASTEXITCODE -ne 0) { throw 'R: is occupied by another directory; it was not changed.' }
  Set-Location R:\
  if ([System.IO.Path]::IsPathRooted($EnvironmentPath) -or $EnvironmentPath.Contains('..')) {
    throw 'EnvironmentPath must name a relative project directory'
  }
  $python = Join-Path "R:\$EnvironmentPath" 'Scripts/python.exe'
  if (-not (Test-Path $python)) {
    & $base -m venv --without-pip $EnvironmentPath
    if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed' }
    & $python -m ensurepip --upgrade --default-pip | Out-Host
    if ($LASTEXITCODE -ne 0) { throw 'Short-path pip bootstrap failed; partial environment was retained' }
    $InstallDependencies = $true
  }
  & $python -B -c 'import sys; print(sys.version); sys.exit(0 if sys.version_info[:2] == (3,11) else 1)' | Out-Host
  if ($LASTEXITCODE -ne 0) { throw 'Existing environment is not Python 3.11; choose a new EnvironmentPath. Nothing was replaced.' }
  if ($InstallDependencies) {
    & $python -m pip install -e '.[dev]' | Out-Host
    if ($LASTEXITCODE -ne 0) { throw 'Declared dependency installation failed' }
  }
  return [PSCustomObject]@{ Python = $python; Root = 'R:\' }
}
