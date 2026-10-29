package hack.sber.service;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.server.ResponseStatusException;

import java.io.IOException;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDate;
import java.time.format.DateTimeParseException;
import java.util.Set;
import java.util.TreeSet;

@Service
public class CsvValidationService {

    private static final Set<String> SALES_COLUMNS = Set.of(
            "date", "composite_division", "cities_name",
            "warehouse_name", "category_lvl_1",
            "category_lvl_2", "category_lvl_3",
            "article", "characteristic", "quantity",
            "sales_sum", "cost_price", "discount_sum",
            "volume_weight"
    );

    private static final Set<String> STOCK_COLUMNS = Set.of(
            "date", "warehouse_name", "char_article", "stock"
    );

    public LocalDate validateSales(Path file) throws IOException {
        TreeSet<LocalDate> dates = new TreeSet<>();

        try (Reader reader = Files.newBufferedReader(
                file, StandardCharsets.UTF_8);
             CSVParser parser = CSVFormat.DEFAULT.builder()
                     .setHeader()
                     .setSkipHeaderRecord(true)
                     .get()
                     .parse(reader)) {

            validateColumns(parser.getHeaderMap().keySet(),
                    SALES_COLUMNS, "sales");

            for (CSVRecord row : parser) {
                try {
                    dates.add(LocalDate.parse(row.get("date")));
                } catch (DateTimeParseException |
                         IllegalArgumentException e) {
                    throw new ResponseStatusException(
                            HttpStatus.UNPROCESSABLE_ENTITY,
                            "Некорректная дата в sales: строка "
                                    + row.getRecordNumber()
                    );
                }
            }
        }

        if (dates.isEmpty()) {
            throw new ResponseStatusException(
                    HttpStatus.UNPROCESSABLE_ENTITY,
                    "Файл sales не содержит данных"
            );
        }

        LocalDate first = dates.first();
        LocalDate last = dates.last();

        long days = last.toEpochDay() - first.toEpochDay() + 1;

        if (days < 180) {
            throw new ResponseStatusException(
                    HttpStatus.UNPROCESSABLE_ENTITY,
                    "История продаж должна охватывать минимум 180 дней"
            );
        }

        return last;
    }

    public void validateStock(Path file) throws IOException {
        try (Reader reader = Files.newBufferedReader(
                file, StandardCharsets.UTF_8);
             CSVParser parser = CSVFormat.DEFAULT.builder()
                     .setHeader()
                     .setSkipHeaderRecord(true)
                     .get()
                     .parse(reader)) {

            validateColumns(parser.getHeaderMap().keySet(),
                    STOCK_COLUMNS, "stock");

            if (!parser.iterator().hasNext()) {
                throw new ResponseStatusException(
                        HttpStatus.UNPROCESSABLE_ENTITY,
                        "Файл stock не содержит данных"
                );
            }
        }
    }

    private void validateColumns(
            Set<String> actual,
            Set<String> required,
            String fileName
    ) {
        if (!actual.containsAll(required)) {
            Set<String> missing = new java.util.HashSet<>(required);
            missing.removeAll(actual);

            throw new ResponseStatusException(
                    HttpStatus.UNPROCESSABLE_ENTITY,
                    "В файле " + fileName
                            + " отсутствуют колонки: " + missing
            );
        }
    }
}