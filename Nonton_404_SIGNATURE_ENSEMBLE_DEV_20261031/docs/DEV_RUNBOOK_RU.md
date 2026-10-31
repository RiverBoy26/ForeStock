# Runbook для разработчиков

## Что считать исходным эталоном

`forecasts/v81_catboost.csv` — неизменяемый эталон `прогноз_5` для конкурса. `v82_confidence_amount.csv` и `v83_*.csv` — отдельные кандидаты. Не заменять эталон экспериментом при импорте в PostgreSQL.

## Верификация

Запускать из корня архива в PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\install_windows.ps1
powershell -ExecutionPolicy Bypass -File .\verify_windows.ps1
.\.venv\Scripts\python.exe -m app.catboost_inference --verify --recompute-base
powershell -ExecutionPolicy Bypass -File .\run_v83_windows.ps1 -Variant nearest -Verify
powershell -ExecutionPolicy Bypass -File .\run_v83_windows.ps1 -Variant smape -Verify
.\.venv\Scripts\python.exe -m unittest tests.test_v83 -v
```

## Повторная настройка экспериментальной головы (не затрагивает эталон)

```powershell
.\.venv\Scripts\python.exe -m app.v82_retrain_positive --iterations 350 --threads 8 --out .\refit_positive\conditional_positive_quantity.cbm
```

Файл `refit_positive` **не активируется автоматически**. Поменять веса и признать кандидата улучшением можно только после повторного исторического backtest.

## Формат интеграции

Каждый CSV: `week,warehouse_name,article,quantity`. Важно хранить `model_version` отдельно от ключей таблицы, чтобы сравнивать v8.1/v8.2/v8.3 без перезаписи. `quantity` не округлять в Java по умолчанию: округление — часть **конкретного** варианта v8.3, а не требование API или формата.

Встроенное `app/api.py` отдаёт только v8.1; для выбора v8.2/v8.3 в приложении нужен импорт разных CSV или реализация версии API. Для инференса на новых файлах придётся разработать отдельный полный pipeline; текущий frozen snapshot нельзя использовать в качестве данных нового пользователя.

## Проверки на backend

- Полный набор ожидаемых 79 101 ключей, без дубликатов, пропусков и нечисловых значений.
- `week` — понедельники девяти недель с 2026-07-27.
- `quantity >= 0`, float разрешён; данные о будущем в истории не используются.
- Не выводить файл данных или модельные OOF в публичные логи и репозитории.
- API ключ из переменной окружения, не из фронтенда.

## Каталог кода

`app/catboost_inference.py` — исходная v8.1; `app/v82_inference.py` — новый условный регрессор; `app/v83_inference.py` — 2 варианта округления; `app/inference.py` — сквозное применение 39 LightGBM; `app/v82_retrain_positive.py` — локальное переобучение новой головы; `app/api.py` — read-only API.
