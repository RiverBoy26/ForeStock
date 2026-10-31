param(
 [switch]$Refit,
 [switch]$Forward,
 [int]$Threads = 4
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$Python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (!(Test-Path $Python)) { throw 'No .venv environment. First run: powershell -ExecutionPolicy Bypass -File .\install_windows.ps1' }
& $Python -m app.catboost_inference --verify
if ($LASTEXITCODE -ne 0) { throw 'Original CatBoost replay failed' }
if (-not $Refit) {
  & $Python -m app.v82_inference --verify
  if ($LASTEXITCODE -ne 0) { throw 'Bundled experimental reference verification failed' }
}
if ($Refit) {
  & $Python -m app.v82_retrain_positive --threads $Threads --out .\models\experimental\conditional_positive_quantity.cbm
  if ($LASTEXITCODE -ne 0) { throw 'Experimental positive model refit failed' }
}
& $Python -m app.v82_inference --out .\forecasts\v82_confidence_amount.csv
if ($LASTEXITCODE -ne 0) { throw 'V82 prediction failed' }
& $Python -m app.v82_inference --verify
if ($LASTEXITCODE -ne 0) { throw 'V82 verification failed' }
& $Python -m unittest discover -s tests -p test_v82.py -v
if ($LASTEXITCODE -ne 0) { throw 'V82 unit tests failed' }
if ($Forward) {
  & $Python -m app.v82_forward_test --threads $Threads --out .\research_v82\reproduced_forward.csv
  if ($LASTEXITCODE -ne 0) { throw 'V82 historical forward validation failed' }
}
Write-Host 'SUCCESS: original and experimental forecasts produced, archive untouched.' -ForegroundColor Green
