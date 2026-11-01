package hack.sber.model;

import jakarta.persistence.*;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.UUID;


@Entity
@Table(name = "forecast_datasets")
public class DatasetRecord {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private UUID id;

    @Column(nullable = false)
    private String salesFilePath;

    @Column(nullable = false)
    private String stockFilePath;

    @Column(nullable = false)
    private LocalDate lastSalesDate;

    @Column(nullable = false)
    private LocalDateTime createdAt;

    private String forecastFilePath;

    private Integer forecastStartWeek;

    private Integer forecastEndWeek;

    private LocalDateTime forecastCreatedAt;

    @Column(columnDefinition = "TEXT")
    private String analyticsJson;

    protected DatasetRecord() {
    }

    public DatasetRecord(
            String salesFilePath,
            String stockFilePath,
            LocalDate lastSalesDate
    ) {
        this.salesFilePath = salesFilePath;
        this.stockFilePath = stockFilePath;
        this.lastSalesDate = lastSalesDate;
        this.createdAt = LocalDateTime.now();
    }

    public UUID getId() {
        return id;
    }

    public String getSalesFilePath() {
        return salesFilePath;
    }

    public String getStockFilePath() {
        return stockFilePath;
    }

    public LocalDate getLastSalesDate() {
        return lastSalesDate;
    }

    public LocalDate getCutoffDate() {
        return lastSalesDate;
    }

    public LocalDateTime getCreatedAt() {
        return createdAt;
    }

    public String getForecastFilePath() {
        return forecastFilePath;
    }

    public Integer getForecastStartWeek() {
        return forecastStartWeek;
    }

    public Integer getForecastEndWeek() {
        return forecastEndWeek;
    }

    public LocalDateTime getForecastCreatedAt() {
        return forecastCreatedAt;
    }

    public String getAnalyticsJson() {
        return analyticsJson;
    }

    public void completeForecast(
            String forecastFilePath,
            int forecastStartWeek,
            int forecastEndWeek,
            LocalDateTime forecastCreatedAt,
            String analyticsJson
    ) {
        this.forecastFilePath = forecastFilePath;
        this.forecastStartWeek = forecastStartWeek;
        this.forecastEndWeek = forecastEndWeek;
        this.forecastCreatedAt = forecastCreatedAt;
        this.analyticsJson = analyticsJson;
    }
}
