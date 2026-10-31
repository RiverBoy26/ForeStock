$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not $env:NONTON_API_KEY) { throw "Set NONTON_API_KEY first" }
& .\.venv\Scripts\python.exe -m uvicorn app.api:api --host 127.0.0.1 --port 8080
