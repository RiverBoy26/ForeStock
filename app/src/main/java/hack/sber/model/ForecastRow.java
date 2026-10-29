package hack.sber.model;

import java.math.BigDecimal;
import java.time.LocalDate;

public record ForecastRow(
        LocalDate date,
        String warehouse_name,
        String article,
        BigDecimal quantity
) {
}