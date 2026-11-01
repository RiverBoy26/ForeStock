package hack.sber.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import hack.sber.model.ForecastAnalyticsSnapshot;
import hack.sber.repository.DatasetRepository;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDate;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.mock;

class ForecastAnalyticsServiceTest {

    @TempDir
    Path temporaryDirectory;

    @Test
    void aggregatesLatestForecastStatisticsAndWeeklySeries() throws Exception {
        Path sales = temporaryDirectory.resolve("sales.csv");
        Files.writeString(sales, """
                date,warehouse_name,article,quantity
                2026-07-13,warehouse_1,article_1,2
                2026-07-14,warehouse_1,article_2,3
                2026-07-20,warehouse_2,article_1,5
                """, StandardCharsets.UTF_8);

        Path stock = temporaryDirectory.resolve("stock.csv");
        Files.writeString(stock, """
                date,warehouse_name,article,stock
                2026-07-19,warehouse_1,article_1,100
                2026-07-26,warehouse_1,article_1,7
                2026-07-26,warehouse_2,article_2,11
                """, StandardCharsets.UTF_8);

        String forecast = """
                week,warehouse_name,article,quantity
                2026-07-27,warehouse_1,article_1,4
                2026-07-27,warehouse_2,article_2,6
                2026-08-03,warehouse_1,article_1,8
                """;

        ForecastAnalyticsService service = new ForecastAnalyticsService(
                mock(DatasetRepository.class), new ObjectMapper());
        ForecastAnalyticsSnapshot result = service.calculate(
                sales, stock, forecast, LocalDate.parse("2026-07-26"));

        assertEquals(2, result.articleCount());
        assertEquals(2, result.warehouseCount());
        assertEquals(10, result.soldUnits());
        assertEquals(18, result.endingStock());
        assertEquals(2, result.historical().size());
        assertEquals(5, result.historical().get(0).value());
        assertEquals(5, result.historical().get(1).value());
        assertEquals(2, result.forecast().size());
        assertEquals(10, result.forecast().get(0).value());
        assertEquals(8, result.forecast().get(1).value());
    }
}
