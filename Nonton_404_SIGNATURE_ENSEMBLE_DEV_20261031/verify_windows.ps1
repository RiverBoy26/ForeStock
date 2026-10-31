$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
& .\.venv\Scripts\python.exe -m app.catboost_inference --verify --recompute-base
if ($LASTEXITCODE -ne 0) { throw "Inference verification failed" }
