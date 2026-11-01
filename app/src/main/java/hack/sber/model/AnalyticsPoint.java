package hack.sber.model;

import java.time.LocalDate;

public record AnalyticsPoint(
        LocalDate date,
        double value
) {
}
