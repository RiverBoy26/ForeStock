param(
  [ValidateSet('smoke','quick','full')][string]$Mode = 'full',
  [string]$RawZip = '.\hakaton.zip',
  [string]$Out = '.\results_NEW_FULL',
  [int]$Threads = 8
)
$ErrorActionPreference = 'Stop'
$python = '.\.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw 'Run install_windows.ps1 first' }
if (-not (Test-Path $RawZip)) { throw "Missing ${RawZip}: provide original source ZIP" }
& $python -u train.py --raw-zip $RawZip --reference '.\reference\official_forecast5.csv' --out $Out --mode $Mode --threads $Threads
if ($LASTEXITCODE -ne 0) { throw "Training failed: $LASTEXITCODE" }
