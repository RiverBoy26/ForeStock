$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
python -m app.inference
if ($LASTEXITCODE -ne 0) { throw 'Inference failed' }
python -m app.validate_release
if ($LASTEXITCODE -ne 0) { throw 'Validation failed' }
