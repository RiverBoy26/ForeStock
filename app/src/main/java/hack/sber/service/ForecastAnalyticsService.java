package hack.sber.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import hack.sber.model.AnalyticsPoint;
import hack.sber.model.DatasetRecord;
import hack.sber.model.ForecastAnalyticsResponse;
import hack.sber.model.ForecastAnalyticsSnapshot;
import hack.sber.repository.DatasetRepository;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.server.ResponseStatusException;

import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.DayOfWeek;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.temporal.TemporalAdjusters;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.TreeMap;

@Service
public class ForecastAnalyticsService {

    private static final int HISTORY_WEEKS_ON_CHART = 12;

    private final DatasetRepository repository;
    private final ObjectMapper objectMapper;

    public ForecastAnalyticsService(
            DatasetRepository repository,
            ObjectMapper objectMapper
    ) {
        this.repository = repository;
        this.objectMapper = objectMapper;
    }

    public void completeForecast(
            DatasetRecord dataset,
            Path salesPath,
            Path stockPath,
            String forecastCsv,
            int startWeek,
            int endWeek
    ) throws IOException {
        ForecastAnalyticsSnapshot snapshot = calculate(
                salesPath, stockPath, forecastCsv, dataset.getCutoffDate());
        String analyticsJson = objectMapper.writeValueAsString(snapshot);

        Path forecastPath = salesPath.getParent().resolve(
                "forecast-weeks-" + startWeek + "-" + endWeek + ".csv");
        Files.writeString(forecastPath, forecastCsv, StandardCharsets.UTF_8);

        dataset.completeForecast(
                forecastPath.toString(),
                startWeek,
                endWeek,
                LocalDateTime.now(),
                analyticsJson
        );
        repository.save(dataset);
    }

    public ForecastAnalyticsResponse getLatest() {
        DatasetRecord dataset = repository
                .findTopByAnalyticsJsonIsNotNullOrderByForecastCreatedAtDesc()
                .orElseThrow(() -> new ResponseStatusException(
                        HttpStatus.NOT_FOUND,
                        "Успешных прогнозов пока нет"
                ));

        try {
            ForecastAnalyticsSnapshot snapshot = objectMapper.readValue(
                    dataset.getAnalyticsJson(), ForecastAnalyticsSnapshot.class);
            return new ForecastAnalyticsResponse(
                    repository.countByAnalyticsJsonIsNotNull(),
                    dataset.getForecastCreatedAt(),
                    snapshot.articleCount(),
                    snapshot.warehouseCount(),
                    snapshot.soldUnits(),
                    snapshot.endingStock(),
                    snapshot.historical(),
                    snapshot.forecast()
            );
        } catch (JsonProcessingException e) {
            throw new ResponseStatusException(
                    HttpStatus.INTERNAL_SERVER_ERROR,
                    "Не удалось прочитать аналитику последнего прогноза",
                    e
            );
        }
    }

    ForecastAnalyticsSnapshot calculate(
            Path salesPath,
            Path stockPath,
            String forecastCsv,
            LocalDate cutoffDate
    ) throws IOException {
        Set<String> articles = new HashSet<>();
        Set<String> warehouses = new HashSet<>();
        TreeMap<LocalDate, Double> historical = new TreeMap<>();
        double soldUnits = 0;

        try (Reader reader = Files.newBufferedReader(
                salesPath, StandardCharsets.UTF_8);
             CSVParser parser = csvParser(reader)) {
            for (CSVRecord row : parser) {
                LocalDate date = LocalDate.parse(row.get("date"));
                if (date.isAfter(cutoffDate)) {
                    continue;
                }
                double quantity = Double.parseDouble(row.get("quantity"));
                soldUnits += quantity;
                articles.add(row.get("article"));
                warehouses.add(row.get("warehouse_name"));
                LocalDate week = date.with(
                        TemporalAdjusters.previousOrSame(DayOfWeek.MONDAY));
                historical.merge(week, quantity, Double::sum);
            }
        }

        LocalDate latestStockDate = null;
        double endingStock = 0;
        try (Reader reader = Files.newBufferedReader(
                stockPath, StandardCharsets.UTF_8);
             CSVParser parser = csvParser(reader)) {
            for (CSVRecord row : parser) {
                LocalDate date = LocalDate.parse(row.get("date"));
                if (date.isAfter(cutoffDate)) {
                    continue;
                }
                if (latestStockDate == null || date.isAfter(latestStockDate)) {
                    latestStockDate = date;
                    endingStock = 0;
                }
                if (date.equals(latestStockDate)) {
                    endingStock += Double.parseDouble(row.get("stock"));
                }
            }
        }

        TreeMap<LocalDate, Double> forecast = new TreeMap<>();
        try (Reader reader = new StringReader(forecastCsv);
             CSVParser parser = csvParser(reader)) {
            for (CSVRecord row : parser) {
                LocalDate week = LocalDate.parse(row.get("week"));
                double quantity = Double.parseDouble(row.get("quantity"));
                forecast.merge(week, quantity, Double::sum);
            }
        }

        return new ForecastAnalyticsSnapshot(
                articles.size(),
                warehouses.size(),
                soldUnits,
                endingStock,
                lastPoints(historical, HISTORY_WEEKS_ON_CHART),
                points(forecast)
        );
    }

    private CSVParser csvParser(Reader reader) throws IOException {
        return CSVFormat.DEFAULT.builder()
                .setHeader()
                .setSkipHeaderRecord(true)
                .get()
                .parse(reader);
    }

    private List<AnalyticsPoint> lastPoints(
            TreeMap<LocalDate, Double> values,
            int limit
    ) {
        List<AnalyticsPoint> all = points(values);
        return new ArrayList<>(all.subList(
                Math.max(0, all.size() - limit), all.size()));
    }

    private List<AnalyticsPoint> points(TreeMap<LocalDate, Double> values) {
        return values.entrySet().stream()
                .map(entry -> new AnalyticsPoint(entry.getKey(), entry.getValue()))
                .toList();
    }
}
