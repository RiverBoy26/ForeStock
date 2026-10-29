package hack.sber.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.FileSystemResource;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.web.util.UriBuilder;

import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Writer;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.time.Duration;
import java.util.function.Function;

@Component
public class NontonPredictionEngine implements PredictionEngine {

    private final RestClient client;
    private final String apiKey;

    public NontonPredictionEngine(
            RestClient.Builder builder,
            @Value("${nonton.base-url:http://127.0.0.1:8080}") String baseUrl,
            @Value("${nonton.api-key:}") String apiKey,
            @Value("${nonton.connect-timeout:10s}") Duration connectTimeout,
            @Value("${nonton.read-timeout:30m}") Duration readTimeout
    ) {
        var requestFactory = new SimpleClientHttpRequestFactory();
        requestFactory.setConnectTimeout(connectTimeout);
        requestFactory.setReadTimeout(readTimeout);
        this.client = builder
                .baseUrl(baseUrl)
                .requestFactory(requestFactory)
                .build();
        this.apiKey = apiKey;
    }

    @Override
    public void predict(
            Path salesFile,
            Path stockFile,
            String article,
            String warehouse,
            String variant,
            Writer output
    ) throws IOException {
        requestPrediction(salesFile, stockFile, article, warehouse, variant, output);
    }

    @Override
    public void predictAll(
            Path salesFile,
            Path stockFile,
            String variant,
            Writer output
    ) throws IOException {
        requestPrediction(salesFile, stockFile, null, null, variant, output);
    }

    private void requestPrediction(
            Path salesFile,
            Path stockFile,
            String article,
            String warehouse,
            String variant,
            Writer output
    ) throws IOException {
        var multipart = new MultipartBodyBuilder();
        multipart.part("sales", new FileSystemResource(salesFile));
        multipart.part("stock", new FileSystemResource(stockFile));

        Function<UriBuilder, URI> uri = builder -> {
            builder.path("/v1/predict").queryParam("variant", variant);
            if (article != null && !article.isBlank()) {
                builder.queryParam("article", article);
            }
            if (warehouse != null && !warehouse.isBlank()) {
                builder.queryParam("warehouse_name", warehouse);
            }
            return builder.build();
        };

        try {
            client.post()
                    .uri(uri)
                    .headers(headers -> {
                        if (apiKey != null && !apiKey.isBlank()) {
                            headers.set("X-API-Key", apiKey);
                        }
                    })
                    .contentType(MediaType.MULTIPART_FORM_DATA)
                    .body(multipart.build())
                    .exchange((request, response) -> {
                        if (response.getStatusCode().isError()) {
                            String detail = new String(
                                    response.getBody().readAllBytes(),
                                    StandardCharsets.UTF_8
                            );
                            HttpStatus status = switch (response.getStatusCode().value()) {
                                case 413 -> HttpStatus.PAYLOAD_TOO_LARGE;
                                case 422 -> HttpStatus.UNPROCESSABLE_ENTITY;
                                default -> HttpStatus.BAD_GATEWAY;
                            };
                            throw new ResponseStatusException(
                                    status,
                                    "Ошибка ML-сервиса: " + detail
                            );
                        }
                        try (var reader = new InputStreamReader(
                                response.getBody(), StandardCharsets.UTF_8)) {
                            reader.transferTo(output);
                        }
                        return null;
                    });
        } catch (ResponseStatusException e) {
            throw e;
        } catch (Exception e) {
            throw new ResponseStatusException(
                    HttpStatus.BAD_GATEWAY,
                    "ML-сервис недоступен или завершил запрос с ошибкой",
                    e
            );
        }
    }
}
