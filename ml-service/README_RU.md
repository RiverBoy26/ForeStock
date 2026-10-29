# Nonton v7.4 — MODEL PACKAGE / Reference Production Service

**Назначение:** готовый для интеграции артефакт модели прогнозирования `quantity` на 9 недель **27.07.2026–27.09.2026**, по 8 789 парам `warehouse_name × article`. Для Java/Spring Boot или другого backend есть отдельный Python inference сервис с HTTP API и Docker. Это **зафиксированный срез** модели и исторических признаков на cutoff **26.07.2026**; для более поздних дат требуется новый ETL, новые признаки, валидация и переобучение.

## Содержимое

- `models/base/*.txt` — **39 реально обученных LightGBM** (глобальные Tweedie/Poisson; модели складов и групп; recursive). `models/meta_sale_probability.txt` и `models/meta_conditional_quantity.txt` — **ещё 2 META-модели**. **Итого 41 обученная модель.**
- `snapshots/weekly_history_2026-07-26.npz` — компактный исторический feature snapshot (без исходного hakaton.zip). Включает **коммерчески чувствительные агрегированные данные**: ограничьте доступ.
- `app/nonton_v7.py`, `app/nonton_v74.py` — оригинальный feature engineering и обучение, пригодные для дальнейшей доработки.
- `app/inference.py` — **восстанавливает прогноз из 41 сохранённой модели**, не обучая их заново.
- `app/api.py` — FastAPI (read-only) с OpenAPI-документацией, API-key, пагинацией, фильтрами и CSV экспортом.
- `forecasts/selected.csv` — **главная отправка v7.4**, выбранная по пяти forward-проверкам.
- `forecasts/conservative.csv` — альтернативный META-порог `0.575` без дополнительного изменения объёма (экспериментальный кандидат).
- `forecasts/smape_experimental.csv` — вариант с усиленной до 30% поправкой объёма; на forward-проверках дал меньшую среднюю SMAPE, но уступил выбранному варианту по совокупной оценке.
- `forecasts/baseline.csv` — контроль v7 FULL без META.
- `model_manifest.json` — версии, семантика и SHA-256 моделей и snapshot; `app.validate_release` сверяет хеши и ключи.
- `docs/` — архитектура, контракты, модельная карточка, метрики и ограничения. `examples/java` — пример интеграции с Spring Boot.
- `Dockerfile`, `docker-compose.yml` — референс развёртывания, API слушает `127.0.0.1:8080` в хосте.

## Быстрый старт Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m app.validate_release
.\.venv\Scripts\python.exe -m app.inference
$env:NONTON_API_KEY="CHANGE_ME_LONG_RANDOM_SECRET"
.\.venv\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8080
```

Откройте `http://127.0.0.1:8080/docs` (Swagger) и используйте `X-API-Key` для запросов.

```powershell
$headers = @{ 'X-API-Key' = $env:NONTON_API_KEY }
Invoke-RestMethod -Uri 'http://127.0.0.1:8080/v1/model' -Headers $headers
Invoke-RestMethod -Uri 'http://127.0.0.1:8080/v1/forecasts?variant=selected&limit=3' -Headers $headers
# 1 товар, 1 склад, 1 неделя: query URL-encode всех пользовательских значений!
```

### Docker

```powershell
Copy-Item .env.example .env
# обязательно отредактировать секрет в .env
# docker compose up -d --build
```

**В production** разместить API за собственным backend/gateway с TLS, аутентификацией, rate limit, аудитом, мониторингом, контролем доступа к модели/snapshot, резервными копиями и CI/CD. FastAPI здесь — **референсный read-only inference microservice**, а не готовая полностью защищённая корпоративная платформа. Нельзя открывать Docker host port в Интернет напрямую.

## Воспроизводимость прогноза

```powershell
.\.venv\Scripts\python.exe -m app.inference --out forecasts
.\.venv\Scripts\python.exe -m app.verify_inference
.\.venv\Scripts\python.exe -m app.validate_release
.\.venv\Scripts\python.exe -m pytest tests -q
```

Проверено на исходных сохранённых моделях: **`selected.csv` и `baseline.csv` полностью совпадают по числам и порядку строк с результатами полного обучения** (79 101 строка). CSV-сериализация может отличаться форматированием разрядов, значения идентичны.

Для нового конкурса/даты нужна отдельная архитектура: регулярная загрузка новых фактов, версии feature store и справочников, соблюдение cutoff, периодическая валидация и новая регистрация модели. **Данные после 26.07.2026 не входили в обучение и snapshot.** `composite_division` здесь департамент/прокси, а не подтверждённый канал продаж.

## Варианты и их валидация

| Вариант | Логика | Позиционирование |
|---|---|---|
| `selected` | META порог 0.55 + 15% коррекции объёма при активности ≥7 недель из 13 | основной, выбран по пяти историческим окнам |
| `conservative` | META порог 0.575, без коррекции объёма | экспериментальный, более осторожный |
| `smape_experimental` | META порог 0.55 + 30% коррекции при активности ≥7/13 | дополнительный эксперимент по SMAPE |
| `baseline` | исходный гибрид SIGNATURE + M5-lite, порог 1.0 | контрольная версия |

Официальные метрики 27.07–27.09 неизвестны. Четыре CSV — **четыре стратегии**, а не три подтверждённо лучших ответа. Многократно изученный майский holdout не является независимым конкурсным тестом.

Контакт между командами разработки: см. `docs/ARCHITECTURE_RU.md` и `docs/MODEL_CARD_RU.md`. Версии всех файлов фиксируются в `model_manifest.json`.
