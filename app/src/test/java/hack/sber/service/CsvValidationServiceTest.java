package hack.sber.service;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.web.server.ResponseStatusException;

import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDate;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CsvValidationServiceTest {

    private static final String SALES_HEADER = String.join(",",
            "date", "composite_division", "cities_name", "warehouse_name",
            "category_lvl_1", "category_lvl_2", "category_lvl_3",
            "article", "characteristic", "quantity", "sales_sum",
            "discount_sum");

    private final CsvValidationService service = new CsvValidationService();

    @TempDir
    Path temporaryDirectory;

    @Test
    void preservesBlankAmountFieldsInsideRetainedWindow() throws Exception {
        Path file = write("sales.csv", SALES_HEADER + "\n"
                + salesRow("2025-07-27", "1", "", "") + "\n"
                + salesRow("2025-07-28", "1", "", "") + "\n");

        service.retainDateRange(file,
                LocalDate.parse("2025-07-28"),
                LocalDate.parse("2026-07-26"),
                "sales");

        try (Reader reader = Files.newBufferedReader(file, StandardCharsets.UTF_8);
             CSVParser parser = CSVFormat.DEFAULT.builder()
                     .setHeader()
                     .setSkipHeaderRecord(true)
                     .get()
                     .parse(reader)) {
            CSVRecord row = parser.iterator().next();
            assertEquals("2025-07-28", row.get("date"));
            assertEquals("", row.get("sales_sum"));
            assertEquals("", row.get("discount_sum"));
        }
    }

    @Test
    void acceptsBlankSalesSumAndPreservesItForModelMissingness() throws Exception {
        Path file = write("sales.csv", SALES_HEADER + "\n"
                + salesRow("2025-07-28", "1", "12.5", "0") + "\n"
                + salesRow("2025-07-29", "1", "", "0") + "\n");

        service.retainDateRange(file,
                LocalDate.parse("2025-07-28"),
                LocalDate.parse("2026-07-26"),
                "sales");

        try (Reader reader = Files.newBufferedReader(file, StandardCharsets.UTF_8);
             CSVParser parser = CSVFormat.DEFAULT.builder()
                     .setHeader()
                     .setSkipHeaderRecord(true)
                     .get()
                     .parse(reader)) {
            assertEquals("", parser.getRecords().get(1).get("sales_sum"));
        }
    }

    @Test
    void rejectsNonFiniteAndNegativeSalesValues() throws Exception {
        Path file = write("sales.csv", SALES_HEADER + "\n"
                + salesRow("2025-07-28", "-1", "NaN", "Infinity") + "\n");

        ResponseStatusException error = assertThrows(
                ResponseStatusException.class,
                () -> service.retainDateRange(file,
                        LocalDate.parse("2025-07-28"),
                        LocalDate.parse("2026-07-26"),
                        "sales"));

        assertTrue(error.getReason().contains("quantity"));
        assertTrue(error.getReason().contains("sales_sum"));
        assertTrue(error.getReason().contains("discount_sum"));
    }

    @Test
    void reportsInvalidStockValue() throws Exception {
        Path file = write("stock.csv",
                "date,warehouse_name,article,stock\n"
                        + "2025-07-28,warehouse_name_0,char_article_0,not-a-number\n");

        ResponseStatusException error = assertThrows(
                ResponseStatusException.class,
                () -> service.retainDateRange(file,
                        LocalDate.parse("2025-07-28"),
                        LocalDate.parse("2026-07-26"),
                        "stock"));

        assertTrue(error.getReason().contains("stock"));
        assertTrue(error.getReason().contains("2"));
    }

    @Test
    void preservesMissingTrailingDiscount() throws Exception {
        Path file = write("sales.csv", SALES_HEADER + "\n"
                + salesRow("2025-07-28", "1", "12.5", "")
                        .replaceFirst(",$", "") + "\n");

        service.retainDateRange(file,
                LocalDate.parse("2025-07-28"),
                LocalDate.parse("2026-07-26"),
                "sales");

        try (Reader reader = Files.newBufferedReader(file, StandardCharsets.UTF_8);
             CSVParser parser = CSVFormat.DEFAULT.builder()
                     .setHeader()
                     .setSkipHeaderRecord(true)
                     .get()
                     .parse(reader)) {
            assertEquals("", parser.iterator().next().get("discount_sum"));
        }
    }

    private Path write(String fileName, String contents) throws Exception {
        Path file = temporaryDirectory.resolve(fileName);
        Files.writeString(file, contents, StandardCharsets.UTF_8);
        return file;
    }

    private String salesRow(
            String date,
            String quantity,
            String salesSum,
            String discountSum
    ) {
        return String.join(",",
                date, "division", "city", "warehouse_name_0",
                "category_1", "category_2", "", "char_article_0",
                "characteristic", quantity, salesSum, discountSum);
    }
}
