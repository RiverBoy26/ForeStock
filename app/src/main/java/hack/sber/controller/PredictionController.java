package hack.sber.controller;

import hack.sber.model.DatasetRecord;
import hack.sber.service.PredictionEngine;
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

    private final PredictionService service;
    private final PredictionEngine engine;

    public PredictionController(
            PredictionService service,
            PredictionEngine engine
    ) {
        this.service = service;
        this.engine = engine;
    }

    @PostMapping(consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<byte[]> predict(
            @RequestParam("sales") org.springframework.web.multipart
                    .MultipartFile sales,
            @RequestParam("stock") org.springframework.web.multipart
                    .MultipartFile stock,
            @RequestParam(defaultValue = "selected") String variant
    ) throws Exception {

        DatasetRecord dataset = service.upload(sales, stock);

        try {
            var writer = new StringWriter();
            engine.predict(
                    service.getSalesPath(dataset),
                    service.getStockPath(dataset),
                    null,
                    null,
                    variant,
                    writer
            );
            return csvResponse("forecast.csv", writer.toString());
        } catch (Exception e) {
            service.delete(dataset);
            throw e;
        }
    }

    @GetMapping
    public ResponseEntity<byte[]> predictFiltered(
            @RequestParam(required = false) String article,
            @RequestParam(required = false) String warehouse,
            @RequestParam(defaultValue = "selected") String variant
    ) {
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
                    writer
            );
        } catch (java.io.IOException e) {
            throw new org.springframework.web.server.ResponseStatusException(
                    org.springframework.http.HttpStatus.BAD_GATEWAY,
                    "Не удалось получить прогноз от ML-сервиса",
                    e
            );
        }
        return csvResponse("filtered_forecast.csv", writer.toString());
    }

    @GetMapping("/all")
    public ResponseEntity<byte[]> predictAll(
            @RequestParam(defaultValue = "csv") String format,
            @RequestParam(defaultValue = "selected") String variant
    ) {
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
                    writer
            );
        } catch (java.io.IOException e) {
            throw new org.springframework.web.server.ResponseStatusException(
                    org.springframework.http.HttpStatus.BAD_GATEWAY,
                    "Не удалось получить прогноз от ML-сервиса",
                    e
            );
        }
        return csvResponse("full_forecast.csv", writer.toString());
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
}
