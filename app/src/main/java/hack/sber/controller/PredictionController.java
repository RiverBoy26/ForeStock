package hack.sber.controller;

import hack.sber.model.DatasetRecord;
import hack.sber.service.PredictionEngine;
import hack.sber.service.ForecastAnalyticsService;
import hack.sber.service.PredictionService;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.io.StringWriter;
import java.nio.charset.StandardCharsets;

@RestController
@RequestMapping("/predict")
public class PredictionController {

    private static final int MIN_FORECAST_WEEK = 1;
    private static final int MAX_FORECAST_WEEK = 9;

    private final PredictionService service;
    private final PredictionEngine engine;
    private final ForecastAnalyticsService analyticsService;

    public PredictionController(
            PredictionService service,
            PredictionEngine engine,
            ForecastAnalyticsService analyticsService
    ) {
        this.service = service;
        this.engine = engine;
        this.analyticsService = analyticsService;
    }

    @PostMapping(consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<byte[]> predict(
            @RequestParam("sales") org.springframework.web.multipart
                    .MultipartFile sales,
            @RequestParam("stock") org.springframework.web.multipart
                    .MultipartFile stock,
            @RequestParam(defaultValue = "selected") String variant,
            @RequestParam(name = "start_week", defaultValue = "1") int startWeek,
            @RequestParam(name = "end_week", defaultValue = "9") int endWeek
    ) throws Exception {

        validateForecastInterval(startWeek, endWeek);

        DatasetRecord dataset = service.upload(sales, stock);

        try {
            var writer = new StringWriter();
            engine.predict(
                    service.getSalesPath(dataset),
                    service.getStockPath(dataset),
                    null,
                    null,
                    variant,
                    startWeek,
                    endWeek,
                    dataset.getCutoffDate(),
                    writer
            );
            analyticsService.completeForecast(
                    dataset,
                    service.getSalesPath(dataset),
                    service.getStockPath(dataset),
                    writer.toString(),
                    startWeek,
                    endWeek
            );
            return csvResponse(forecastFilename("forecast", startWeek, endWeek),
                    writer.toString());
        } catch (Exception e) {
            service.delete(dataset);
            throw e;
        }
    }

    @GetMapping
    public ResponseEntity<byte[]> predictFiltered(
            @RequestParam(required = false) String article,
            @RequestParam(required = false) String warehouse,
            @RequestParam(defaultValue = "selected") String variant,
            @RequestParam(name = "start_week", defaultValue = "1") int startWeek,
            @RequestParam(name = "end_week", defaultValue = "9") int endWeek
    ) {
        validateForecastInterval(startWeek, endWeek);
        if (isBlank(article) && isBlank(warehouse)) {
            throw new org.springframework.web.server
                    .ResponseStatusException(
                    org.springframework.http.HttpStatus.BAD_REQUEST,
                    "Укажите article и/или warehouse"
            );
        }

        DatasetRecord dataset = service.getLatestDataset();

        var writer = new StringWriter();
        try {
            engine.predict(
                    service.getSalesPath(dataset),
                    service.getStockPath(dataset),
                    article,
                    warehouse,
                    variant,
                    startWeek,
                    endWeek,
                    dataset.getCutoffDate(),
                    writer
            );
        } catch (java.io.IOException e) {
            throw new org.springframework.web.server.ResponseStatusException(
                    org.springframework.http.HttpStatus.BAD_GATEWAY,
                    "Не удалось получить прогноз от ML-сервиса",
                    e
            );
        }
        return csvResponse(forecastFilename("filtered_forecast", startWeek, endWeek),
                writer.toString());
    }

    @GetMapping("/all")
    public ResponseEntity<byte[]> predictAll(
            @RequestParam(defaultValue = "csv") String format,
            @RequestParam(defaultValue = "selected") String variant,
            @RequestParam(name = "start_week", defaultValue = "1") int startWeek,
            @RequestParam(name = "end_week", defaultValue = "9") int endWeek
    ) {
        validateForecastInterval(startWeek, endWeek);
        if (!"csv".equalsIgnoreCase(format)) {
            throw new org.springframework.web.server
                    .ResponseStatusException(
                    org.springframework.http.HttpStatus.BAD_REQUEST,
                    "На этом этапе поддерживается только format=csv"
            );
        }

        DatasetRecord dataset = service.getLatestDataset();

        var writer = new StringWriter();
        try {
            engine.predictAll(
                    service.getSalesPath(dataset),
                    service.getStockPath(dataset),
                    variant,
                    startWeek,
                    endWeek,
                    dataset.getCutoffDate(),
                    writer
            );
        } catch (java.io.IOException e) {
            throw new org.springframework.web.server.ResponseStatusException(
                    org.springframework.http.HttpStatus.BAD_GATEWAY,
                    "Не удалось получить прогноз от ML-сервиса",
                    e
            );
        }
        return csvResponse(forecastFilename("full_forecast", startWeek, endWeek),
                writer.toString());
    }

    private ResponseEntity<byte[]> csvResponse(
            String filename,
            String csv
    ) {
        return ResponseEntity.ok()
                .contentType(MediaType.parseMediaType("text/csv"))
                .header(
                        HttpHeaders.CONTENT_DISPOSITION,
                        "attachment; filename=\"" + filename + "\""
                )
                .body(csv.getBytes(StandardCharsets.UTF_8));
    }

    private boolean isBlank(String value) {
        return value == null || value.isBlank();
    }

    private void validateForecastInterval(int startWeek, int endWeek) {
        if (startWeek < MIN_FORECAST_WEEK || endWeek > MAX_FORECAST_WEEK
                || startWeek > endWeek) {
            throw new org.springframework.web.server.ResponseStatusException(
                    org.springframework.http.HttpStatus.BAD_REQUEST,
                    "Период прогноза должен находиться в диапазоне от 1 до 9 недель, "
                            + "а начальная неделя не может быть позже конечной"
            );
        }
    }

    private String forecastFilename(String prefix, int startWeek, int endWeek) {
        return prefix + "_weeks_" + startWeek + "-" + endWeek + ".csv";
    }
}
