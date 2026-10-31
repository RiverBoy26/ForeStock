$ErrorActionPreference = 'Stop'
if (-not (Test-Path '.\.venv\Scripts\python.exe')) { py -3 -m venv .venv; if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' } }
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'pip install failed' }
