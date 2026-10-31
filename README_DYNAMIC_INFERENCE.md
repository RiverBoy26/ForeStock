# Динамический прогноз: фактический статус и запуск

## Статус поддержки

Сейчас работающий upload-путь ForeStock использует **Nonton v7.4**, его сохранённые веса и фиксированный reference snapshot. Это не Signature v8.1. Поддерживается только известная модельная сетка и исходный cutoff `2026-07-26`; произвольные склады, артикулы, короткая история и новые даты не поддерживаются. Signature v8.1 и MOE пока доступны только как исследовательские/воспроизводимые пакеты с фиксированной историей, не как динамический путь от новых CSV.

В интерфейсе загружаются два CSV, каждый не больше 500 MB. Файлы должны содержать поля из секции «Формат входных файлов» ниже. Для полного end-to-end запуска нужен Docker Desktop.

## Запуск существующего приложения

Из корня репозитория:

```powershell
Copy-Item .env.example .env
# Для локальной разработки замени NONTON_API_KEY на собственное значение.
docker compose up --build -d
docker compose ps
```

Открой `http://localhost:8080`. Backend health: `http://localhost:8000/health`; ML Swagger: `http://localhost:8082/docs`. Логи: `docker compose logs -f backend ml frontend`. Остановка: `docker compose down`.

## Формат входных файлов

`sales.csv` требует колонки `date,composite_division,cities_name,warehouse_name,category_lvl_1,category_lvl_2,category_lvl_3,characteristic,article,quantity,sales_sum,discount_sum`.

`stock.csv` требует `date,warehouse_name,article,stock` (также поддерживается старое имя `char_article` для артикула). Даты — `YYYY-MM-DD`. Продажи и остатки должны содержать как минимум 52 недели; фактически ML также требует соответствия фиксированной сетке из snapshot. Дубликаты остатков по дате, складу и артикулу суммируются.

## Python API

ML endpoint: `POST http://localhost:8082/v1/predict` с multipart полями `sales`, `stock`. Поддерживаемые варианты: `selected`, `conservative`, `smape_experimental`, `baseline` (это варианты v7.4; они не равны Signature v8.1). Можно указать `cutoff_date=2026-07-26`, `start_week=1`, `end_week=9`, а также `warehouse_name` и `article`. Если `NONTON_API_KEY` задан, добавь заголовок `X-API-Key`.

Пример PowerShell:

```powershell
$headers = @{ 'X-API-Key' = 'local-development-key-change-me' }
$form = @{
  sales = Get-Item .\sales.csv
  stock = Get-Item .\stock.csv
}
Invoke-WebRequest -Method Post `
  -Uri 'http://localhost:8082/v1/predict?variant=selected&cutoff_date=2026-07-26&start_week=1&end_week=9' `
  -Headers $headers -Form $form -OutFile .\forecast.csv
```

Если cutoff опущен, сервис выводит последнее воскресенье, покрываемое обоими входами. Затем сверяет его с поддерживаемым cutoff. Любая другая дата отклоняется HTTP 422 (`MODEL_OUT_OF_DOMAIN`). Frontend/Java путь пока передаёт файлы синхронно и сам cutoff выбрать не позволяет.

## Проверки

```powershell
cd .\ml-service
python -m pytest tests -q
cd ..
docker compose config
```

Команда `pytest` проверяет имеющиеся unit/inference проверки v7.4 и guard единственного cutoff. Полный golden test Signature v8.1, проверка зависимости прогноза от upload, end-to-end job test и MOE golden test не проходят/не реализованы автоматически: для них нужен оригинальный `hakaton.zip` и динамический Signature builder. Golden CSV нельзя подставлять как вход для такого теста.

## Ограничения

- Сохранённые веса и категориальные ID привязаны к известной сетке snapshot; новые значения отклоняются.
- Произвольный cutoff не разрешён: календарные и origin-признаки обучены на фиксированной дате.
- Текущий frontend — Spring Boot + Thymeleaf, а не React. Job queue, отдельные endpoints статуса/результата и сравнение версий не реализованы.
- Архивные CSV прогнозов — воспроизводимые результаты фиксированного пакета, не результат произвольной новой загрузки.
- Полный аудит и блокеры перечислены в [ml-service/docs/DYNAMIC_INFERENCE_AUDIT.md](ml-service/docs/DYNAMIC_INFERENCE_AUDIT.md).
