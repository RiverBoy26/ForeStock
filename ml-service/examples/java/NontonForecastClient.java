package com.example.nonton;

import org.springframework.http.HttpHeaders;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import java.time.Duration;
import java.util.List;

/** Example Spring Boot 3.x adapter. Dependency: spring-boot-starter-webflux.
 * URL and API key MUST come from your secrets/configuration management.
 */
@Component
public class NontonForecastClient {
    private final WebClient client;

    public NontonForecastClient(org.springframework.core.env.Environment env) {
        String baseUrl = env.getRequiredProperty("nonton.forecast.base-url");
        String apiKey = env.getRequiredProperty("nonton.forecast.api-key");
        this.client = WebClient.builder()
                .baseUrl(baseUrl)
                .defaultHeader("X-API-Key", apiKey)
                .build();
    }

    public record Forecast(String week, String warehouse_name, String article, double quantity) {}
    public record ForecastPage(String variant, int total, int offset, int limit,
                               List<Forecast> items) {}

    /** Production calls should add retries/circuit-breaker/metrics at gateway layer. */
    public ForecastPage getForecasts(String week, String warehouse, String article,
                                     int offset, int limit, String variant) {
        return client.get().uri(uri -> {
                    var builder = uri.path("/v1/forecasts")
                        .queryParam("variant", variant)
                        .queryParam("offset", offset)
                        .queryParam("limit", limit);
                    if (week != null) builder.queryParam("week", week);
                    if (warehouse != null) builder.queryParam("warehouse_name", warehouse);
                    if (article != null) builder.queryParam("article", article);
                    return builder.build();
                }).retrieve().bodyToMono(ForecastPage.class)
                .block(Duration.ofSeconds(10));
    }
}
