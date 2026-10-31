param([switch]$Fast)
$ErrorActionPreference = 'Stop'
$python = '.\.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw 'Run install_windows.ps1 first' }
$argsList = @('replay.py', '--verify', '--out', '.\replayed_forecasts')
if (-not $Fast) { $argsList += '--recompute-experts' }
& $python @argsList
if ($LASTEXITCODE -ne 0) { throw "Replay failed: $LASTEXITCODE" }
