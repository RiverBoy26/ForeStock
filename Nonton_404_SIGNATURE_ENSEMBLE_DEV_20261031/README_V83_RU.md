# Nonton v8.3 — две корректировки количества (эксперимент)

Базовая модель: 39 LightGBM + 2 CatBoost (v8.1) и ещё одна CatBoost условного количества (v8.2).
Дополнительного ML-обучения v8.3 **не требует**: изменяется только постобработка положительных прогнозов.
Официальный прогноз №5 сохранён и не перезаписывается.

## Windows

```powershell
powershell -ExecutionPolicy Bypass -File .\install_windows.ps1
powershell -ExecutionPolicy Bypass -File .\run_v83_windows.ps1 -Variant nearest -Verify
powershell -ExecutionPolicy Bypass -File .\run_v83_windows.ps1 -Variant smape -Verify
.\.venv\Scripts\python.exe -m unittest tests.test_v83 -v
```

Проверка всех 39 исходных LightGBM, а не кэша:
```powershell
powershell -ExecutionPolicy Bypass -File .\run_v83_windows.ps1 -Variant nearest -RecomputeBase -Verify
```

## Готовые CSV

- `forecasts/v83_nearest10.csv` — основная консервативная кандидатура (симметричное округление до 10).
- `forecasts/v83_smape10.csv` — альтернативная кандидатура с более сильным фокусом на SMAPE.
- `forecasts/v81_catboost.csv`, `forecasts/v82_confidence_amount.csv` — исходные контрольные точки.

## Подробности

- `RESEARCH_V83_RU.md` — числа и ограничения.
- `research_v83/` — исходники воспроизводимых тестов, пооконные CSV и график.
- **Проверка не гарантирует улучшение официального Score**; метрики локальных исторических окон и скрытого набора разные.
