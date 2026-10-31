$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
python -m uvicorn app.api:app --host 127.0.0.1 --port 8080
