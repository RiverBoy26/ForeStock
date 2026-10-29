package hack.sber.controller;

import hack.sber.model.DatasetRecord;
import hack.sber.service.PredictionEngine;
import hack.sber.service.PredictionService;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.method.annotation
        .StreamingResponseBody;

import java.io.OutputStreamWriter;
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
    public ResponseEntity<StreamingResponseBody> predict(
            @RequestParam("sales") org.springframework.web.multipart
                    .MultipartFile sales,
            @RequestParam("stock") org.springframework.web.multipart
                    .MultipartFile stock
    ) throws Exception {

        DatasetRecord dataset = service.upload(sales, stock);

        StreamingResponseBody body = output -> {
            var writer = new OutputStreamWriter(
                    output, StandardCharsets.UTF_8);

            engine.predict(
                    service.getSalesPath(dataset),
                    service.getStockPath(dataset),
                    null,
                    null,
                    writer
            );

            writer.flush();
        };

        return csvResponse("forecast.csv", body);
    }

    @GetMapping
    public ResponseEntity<StreamingResponseBody> predictFiltered(
            @RequestParam(required = false) String article,
            @RequestParam(required = false) String warehouse
    ) {
        if (isBlank(article) && isBlank(warehouse)) {
            throw new org.springframework.web.server
                    .ResponseStatusException(
                    org.springframework.http.HttpStatus.BAD_REQUEST,
                    "Укажите article и/или warehouse"
            );
        }

        DatasetRecord dataset = service.getLatestDataset();

        StreamingResponseBody body = output -> {
            var writer = new OutputStreamWriter(
                    output, StandardCharsets.UTF_8);

            engine.predict(
                    service.getSalesPath(dataset),
                    service.getStockPath(dataset),
                    article,
                    warehouse,
                    writer
            );

            writer.flush();
        };

        return csvResponse("filtered_forecast.csv", body);
    }

    @GetMapping("/all")
    public ResponseEntity<StreamingResponseBody> predictAll(
            @RequestParam(defaultValue = "csv") String format
    ) {
        if (!"csv".equalsIgnoreCase(format)) {
            throw new org.springframework.web.server
                    .ResponseStatusException(
                    org.springframework.http.HttpStatus.BAD_REQUEST,
                    "На этом этапе поддерживается только format=csv"
            );
        }

        DatasetRecord dataset = service.getLatestDataset();

        StreamingResponseBody body = output -> {
            var writer = new OutputStreamWriter(
                    output, StandardCharsets.UTF_8);

            engine.predictAll(
                    service.getSalesPath(dataset),
                    service.getStockPath(dataset),
                    writer
            );

            writer.flush();
        };

        return csvResponse("full_forecast.csv", body);
    }

    private ResponseEntity<StreamingResponseBody> csvResponse(
            String filename,
            StreamingResponseBody body
    ) {
        return ResponseEntity.ok()
                .contentType(MediaType.parseMediaType("text/csv"))
                .header(
                        HttpHeaders.CONTENT_DISPOSITION,
                        "attachment; filename=\"" + filename + "\""
                )
                .body(body);
    }

    private boolean isBlank(String value) {
        return value == null || value.isBlank();
    }
}