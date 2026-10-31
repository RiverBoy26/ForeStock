package hack.sber.service;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.server.ResponseStatusException;

import java.io.IOException;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.LocalDate;
import java.time.format.DateTimeParseException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

@Service
public class CsvValidationService {

    private static final int MAX_REPORTED_ROWS = 10;

    private static final Set<String> SALES_COLUMNS = Set.of(
            "date", "composite_division", "cities_name",
            "warehouse_name", "category_lvl_1",
            "category_lvl_2", "category_lvl_3",
            "article", "characteristic", "quantity",
            "sales_sum", "discount_sum"
    );

    private static final Set<String> STOCK_COLUMNS = Set.of(
            "date", "warehouse_name", "article", "stock"
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

        if (days < 364) {
            throw new ResponseStatusException(
                    HttpStatus.UNPROCESSABLE_ENTITY,
                    "История продаж должна охватывать минимум 52 недели"
            );
        }

        return last;
    }

    public LocalDate validateStock(Path file) throws IOException {
        LocalDate lastDate = null;

        try (Reader reader = Files.newBufferedReader(
                file, StandardCharsets.UTF_8);
             CSVParser parser = CSVFormat.DEFAULT.builder()
                     .setHeader()
                     .setSkipHeaderRecord(true)
                     .get()
                     .parse(reader)) {

            validateColumns(parser.getHeaderMap().keySet(),
                    STOCK_COLUMNS, "stock");

            for (CSVRecord row : parser) {
                try {
                    LocalDate date = LocalDate.parse(row.get("date"));
                    if (lastDate == null || date.isAfter(lastDate)) {
                        lastDate = date;
                    }
                } catch (DateTimeParseException |
                         IllegalArgumentException e) {
                    throw new ResponseStatusException(
                            HttpStatus.UNPROCESSABLE_ENTITY,
                            "Некорректная дата в stock: строка "
                                    + row.getRecordNumber()
                    );
                }
            }

            if (lastDate == null) {
                throw new ResponseStatusException(
                        HttpStatus.UNPROCESSABLE_ENTITY,
                        "Файл stock не содержит данных"
                );
            }
        }

        return lastDate;
    }

    public void retainDateRange(
            Path file,
            LocalDate startInclusive,
            LocalDate endInclusive,
            String fileName
    ) throws IOException {
        Path temporary = Files.createTempFile(
                file.getParent(), fileName + "-window-", ".csv");
        long retainedRows = 0;
        Map<String, List<Long>> invalidRows = new LinkedHashMap<>();

        try (Reader reader = Files.newBufferedReader(
                file, StandardCharsets.UTF_8);
             CSVParser parser = CSVFormat.DEFAULT.builder()
                     .setHeader()
                     .setSkipHeaderRecord(true)
                     .get()
                     .parse(reader);
             var writer = Files.newBufferedWriter(
                     temporary, StandardCharsets.UTF_8);
             CSVPrinter printer = CSVFormat.DEFAULT.builder()
                     .setHeader(parser.getHeaderNames().toArray(String[]::new))
                     .get()
                     .print(writer)) {

            for (CSVRecord row : parser) {
                LocalDate date = LocalDate.parse(row.get("date"));
                if (!date.isBefore(startInclusive)
                        && !date.isAfter(endInclusive)) {
                    List<String> normalized = normalizeNumericValues(
                            row, fileName, invalidRows);
                    printer.printRecord(normalized);
                    retainedRows++;
                }
            }
        } catch (IOException | RuntimeException e) {
            Files.deleteIfExists(temporary);
            throw e;
        }

        if (retainedRows == 0) {
            Files.deleteIfExists(temporary);
            throw new ResponseStatusException(
                    HttpStatus.UNPROCESSABLE_ENTITY,
                    "В файле " + fileName
                            + " нет данных за последние 52 недели"
            );
        }

        if (!invalidRows.isEmpty()) {
            Files.deleteIfExists(temporary);
            throw new ResponseStatusException(
                    HttpStatus.UNPROCESSABLE_ENTITY,
                    numericValidationMessage(fileName, invalidRows)
            );
        }

        Files.move(temporary, file, StandardCopyOption.REPLACE_EXISTING);
    }

    private List<String> normalizeNumericValues(
            CSVRecord row,
            String fileName,
            Map<String, List<Long>> invalidRows
    ) {
        List<String> values = new ArrayList<>();
        row.forEach(values::add);
        while (values.size() < row.getParser().getHeaderNames().size()) {
            values.add("");
        }
        long csvLine = row.getRecordNumber() + 1;

        if ("sales".equals(fileName)) {
            validateNumber(row, "quantity", csvLine,
                    invalidRows, true);
            validateOptionalNumber(row, "sales_sum", csvLine,
                    invalidRows);
            validateOptionalNumber(row, "discount_sum", csvLine,
                    invalidRows);
        } else if ("stock".equals(fileName)) {
            validateNumber(row, "stock", csvLine,
                    invalidRows, false);
        }

        return values;
    }

    private void validateNumber(
            CSVRecord row,
            String column,
            long csvLine,
            Map<String, List<Long>> invalidRows,
            boolean nonNegative
    ) {
        String value = value(row, column);
        boolean valid = value != null && !value.isBlank();

        if (valid) {
            try {
                double number = Double.parseDouble(value);
                valid = Double.isFinite(number)
                        && (!nonNegative || number >= 0);
            } catch (NumberFormatException e) {
                valid = false;
            }
        }

        if (!valid) {
            List<Long> rows = invalidRows.computeIfAbsent(
                    column, ignored -> new ArrayList<>());
            if (rows.size() < MAX_REPORTED_ROWS) {
                rows.add(csvLine);
            }
        }
    }

    private void validateOptionalNumber(
            CSVRecord row,
            String column,
            long csvLine,
            Map<String, List<Long>> invalidRows
    ) {
        String value = value(row, column);
        if (value == null || value.isBlank()
                || "null".equalsIgnoreCase(value.trim())) {
            return;
        }
        validateNumber(row, column, csvLine, invalidRows, false);
    }

    private String value(CSVRecord row, String column) {
        return row.isSet(column) ? row.get(column) : null;
    }

    private String numericValidationMessage(
            String fileName,
            Map<String, List<Long>> invalidRows
    ) {
        StringBuilder message = new StringBuilder(
                "В файле " + fileName
                        + " найдены пустые или некорректные числовые значения: ");

        boolean first = true;
        for (Map.Entry<String, List<Long>> entry : invalidRows.entrySet()) {
            if (!first) {
                message.append("; ");
            }
            message.append(entry.getKey())
                    .append(" — строки ")
                    .append(entry.getValue());
            first = false;
        }

        return message.toString();
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
