package org.example.sberhak;

import jakarta.servlet.http.HttpServletResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpEntity;
import org.springframework.http.MediaType;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.stereotype.Controller;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.Locale;

@Controller
public class HelloController {

    private static final Logger log = LoggerFactory.getLogger(HelloController.class);
    private static final long MAX_FILE_SIZE = 500L * 1024 * 1024;
    private static final int MIN_FORECAST_WEEK = 1;
    private static final int MAX_FORECAST_WEEK = 9;

    private final RestClient backendClient;

    public HelloController(@Value("${backend.base-url:http://localhost:8000}") String backendUrl) {
        SimpleClientHttpRequestFactory requestFactory = new SimpleClientHttpRequestFactory();
        requestFactory.setConnectTimeout(Duration.ofSeconds(5));
        requestFactory.setReadTimeout(Duration.ofMinutes(10));

        this.backendClient = RestClient.builder()
                .baseUrl(backendUrl)
                .requestFactory(requestFactory)
                .build();
    }

    @GetMapping("/")
    public String index() {
        return "index";
    }

    @PostMapping(value = "/predict", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public void predict(
            @RequestParam("sales") MultipartFile sales,
            @RequestParam("stock") MultipartFile stock,
            @RequestParam(name = "start_week", defaultValue = "1") int startWeek,
            @RequestParam(name = "end_week", defaultValue = "9") int endWeek,
            HttpServletResponse response
    ) throws IOException {
        String validationError = validateFiles(sales, stock);
        if (validationError != null) {
            writeError(response, HttpServletResponse.SC_BAD_REQUEST, validationError);
            return;
        }

        validationError = validateForecastInterval(startWeek, endWeek);
        if (validationError != null) {
            writeError(response, HttpServletResponse.SC_BAD_REQUEST, validationError);
            return;
        }

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("sales", csvPart(sales));
        body.add("stock", csvPart(stock));

        try {
            backendClient.post()
                    .uri(uriBuilder -> uriBuilder
                            .path("/predict")
                            .queryParam("start_week", startWeek)
                            .queryParam("end_week", endWeek)
                            .build())
                    .contentType(MediaType.MULTIPART_FORM_DATA)
                    .body(body)
                    .exchange((request, backendResponse) -> {
                        response.setStatus(backendResponse.getStatusCode().value());

                        MediaType contentType = backendResponse.getHeaders().getContentType();
                        if (contentType != null) {
                            response.setContentType(contentType.toString());
                        }

                        String disposition = backendResponse.getHeaders().getFirst(
                                HttpHeaders.CONTENT_DISPOSITION);
                        if (disposition != null) {
                            response.setHeader(HttpHeaders.CONTENT_DISPOSITION, disposition);
                        }

                        backendResponse.getBody().transferTo(response.getOutputStream());
                        response.flushBuffer();
                        return null;
                    });
        } catch (ResourceAccessException e) {
            log.error("Backend is unavailable", e);
            writeError(response, HttpServletResponse.SC_BAD_GATEWAY,
                    "Бэкенд недоступен. Повторите попытку позже.");
        } catch (RestClientException e) {
            log.error("Backend request failed", e);
            writeError(response, HttpServletResponse.SC_BAD_GATEWAY,
                    "Ошибка при обращении к сервису прогнозирования.");
        }
    }

    private String validateFiles(MultipartFile sales, MultipartFile stock) {
        if (sales == null || sales.isEmpty() || stock == null || stock.isEmpty()) {
            return "Необходимо выбрать оба CSV-файла.";
        }

        if (sales.getSize() > MAX_FILE_SIZE || stock.getSize() > MAX_FILE_SIZE) {
            return "Размер каждого файла не должен превышать 500 МБ.";
        }

        if (!isCsv(sales) || !isCsv(stock)) {
            return "Допустимы только CSV-файлы.";
        }

        return null;
    }

    private String validateForecastInterval(int startWeek, int endWeek) {
        if (startWeek < MIN_FORECAST_WEEK || endWeek > MAX_FORECAST_WEEK) {
            return "Период прогноза должен находиться в диапазоне от 1 до 9 недель.";
        }
        if (startWeek > endWeek) {
            return "Начальная неделя прогноза не может быть позже конечной.";
        }
        return null;
    }

    private boolean isCsv(MultipartFile file) {
        String filename = file.getOriginalFilename();
        return filename != null
                && filename.toLowerCase(Locale.ROOT).endsWith(".csv")
                && !filename.contains("/")
                && !filename.contains("\\");
    }

    private HttpEntity<?> csvPart(MultipartFile file) {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.parseMediaType("text/csv"));
        return new HttpEntity<>(file.getResource(), headers);
    }

    private void writeError(HttpServletResponse response, int status, String message)
            throws IOException {
        if (response.isCommitted()) {
            return;
        }

        response.reset();
        response.setStatus(status);
        response.setContentType(MediaType.TEXT_PLAIN_VALUE);
        response.setCharacterEncoding(StandardCharsets.UTF_8.name());
        response.getWriter().write(message);
        response.flushBuffer();
    }
}
