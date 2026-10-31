param(
    [ValidateSet('nearest','smape')][string]$Variant = 'nearest',
    [switch]$Verify,
    [switch]$RecomputeBase
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$Python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (!(Test-Path $Python)) { throw 'Create .venv first: powershell -ExecutionPolicy Bypass -File .\install_windows.ps1' }
$cmdargs = @('-m','app.v83_inference','--variant',$Variant)
if ($Verify) { $cmdargs += '--verify' }
if ($RecomputeBase) { $cmdargs += '--recompute-base' }
& $Python @cmdargs
if ($LASTEXITCODE -ne 0) { throw ('v8.3 exited with code '+ $LASTEXITCODE) }
