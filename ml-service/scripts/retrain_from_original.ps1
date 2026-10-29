param(
  [string]$RawZip = "hakaton.zip",
  [string]$Output = "results_v74_retrained",
  [int]$Threads = 8
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
python -m app.nonton_v74 --raw-zip $RawZip --out $Output --threads $Threads --origins 0 --rounds 240 --groups 24
if ($LASTEXITCODE -ne 0) { throw 'Training failed' }
Write-Host 'TRAINING FINISHED. Artifacts in ' $Output
Write-Host 'For production, separately validate and version new model outputs before replacing this deployed release.'
