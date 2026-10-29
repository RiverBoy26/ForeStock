# Интеграция с Java 21 + Spring Boot

В `examples/java/NontonForecastClient.java` — пример типизированного WebClient-клиента. Для Spring MVC можно использовать `RestClient` с тем же REST-контрактом.

Добавьте `spring-boot-starter-webflux` для WebClient. В application.yml конфигурация:

```yaml
nonton:
  forecast:
    base-url: ${NONTON_FORECAST_URL:http://127.0.0.1:8080}
    api-key: ${NONTON_FORECAST_API_KEY}
```

Клиент использует `X-API-Key`, фильтры по неделе, складу, артикулу, пагинацию и контролируемый timeout. Обязательно URL-encode параметров: WebClient делает это при построении URI.

Два варианта реализации бизнеса:

1. **Batch импорт:** backend по расписанию получает `/v1/export/selected.csv`, сохраняет 79 101 прогнозов в PostgreSQL с PK `(model_version, week, warehouse_name, article)`, затем быстро отвечает из БД. Самый простой вариант для конкурсной сдачи и небольшого фиксированного горизонта.
2. **Microservice:** backend запрашивает Python `/v1/forecasts`. Уместно при многократной смене сценариев, но есть дополнительный runtime и сетевые зависимости.

В любом случае version-pin модели, валидация полноты ключей, защита от повторного импорта, неизменяемый cutoff и контроль доступа обязательны.

Ни одна из этих схем пока не представляет сам по себе конвейер обучения на новых данных. Для production retraining нужен отдельный job и versioned feature store.
