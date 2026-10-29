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

    public LocalDateTime getCreatedAt() {
        return createdAt;
    }
}