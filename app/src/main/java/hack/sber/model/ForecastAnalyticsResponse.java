package hack.sber.model;

import java.time.LocalDateTime;
import java.util.List;

public record ForecastAnalyticsResponse(
        long totalForecasts,
        LocalDateTime lastCalculation,
        int articleCount,
        int warehouseCount,
        double soldUnits,
        double endingStock,
        List<AnalyticsPoint> historical,
        List<AnalyticsPoint> forecast
) {
}
