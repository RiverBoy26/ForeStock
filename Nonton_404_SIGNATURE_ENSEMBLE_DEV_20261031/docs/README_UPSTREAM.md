# Nonton v8.1 CatBoost — Production / Research release

This package contains actual trained models: **39 LightGBM base + 2 CatBoost**.
The reference output is `forecasts/v81_catboost.csv` (79,101 rows, fixed 9-week horizon).

## Windows 11, PowerShell, Python 3.11

```powershell
powershell -ExecutionPolicy Bypass -File .\install_windows.ps1
powershell -ExecutionPolicy Bypass -File .\verify_windows.ps1
$env:NONTON_API_KEY = "SET_A_LONG_RANDOM_SECRET"
powershell -ExecutionPolicy Bypass -File .\start_api_windows.ps1
```

Open `http://127.0.0.1:8080/docs` (Swagger). Provide header `X-API-Key`.

## Run from trained models

```powershell
.\.venv\Scripts\python.exe -m app.catboost_inference --verify --recompute-base
.\.venv\Scripts\python.exe -m app.catboost_inference --out generated.csv
```

`--recompute-base`: verifies all 39 LightGBM base models + 2 CatBoost heads.
Without it, reuses frozen base outputs from the training run for faster verification.

## Re-training CatBoost heads on the same OOF data

```powershell
.\.venv\Scripts\python.exe -m app.retrain_catboost --iterations 350 --threads 8 --out refitted_catboost
```

This **does not** re-train the 39 base LightGBM models.

**Important**: This release supports only fixed July 27–Sep 27, 2026 contest period. It does not implement dynamic POST(sales, stock) prediction. See `handoff.md`.
