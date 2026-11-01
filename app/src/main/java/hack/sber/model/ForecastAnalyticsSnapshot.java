package hack.sber.model;

import java.util.List;

public record ForecastAnalyticsSnapshot(
        int articleCount,
        int warehouseCount,
        double soldUnits,
        double endingStock,
        List<AnalyticsPoint> historical,
        List<AnalyticsPoint> forecast
) {
}
