package hack.sber.service;

import hack.sber.model.DatasetRecord;
import hack.sber.repository.DatasetRepository;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDate;

@Service
public class PredictionService {

    private final DatasetRepository repository;
    private final CsvValidationService validator;

    private final Path storageDirectory;

    public PredictionService(
            DatasetRepository repository,
            CsvValidationService validator,
            @Value("${forecast.storage:./forecast-data}")
            String storageDirectory
    ) throws IOException {
        this.repository = repository;
        this.validator = validator;
        this.storageDirectory = Path.of(storageDirectory)
                .toAbsolutePath()
                .normalize();

        Files.createDirectories(this.storageDirectory);
    }

    public DatasetRecord upload(
            MultipartFile sales,
            MultipartFile stock
    ) throws IOException {

        if (sales == null || sales.isEmpty()
                || stock == null || stock.isEmpty()) {
            throw new ResponseStatusException(
                    HttpStatus.BAD_REQUEST,
                    "Необходимо передать оба файла: sales и stock"
            );
        }

        Path directory = Files.createTempDirectory(
                storageDirectory, "dataset-");

        Path salesPath = directory.resolve("sales.csv");
        Path stockPath = directory.resolve("stock.csv");

        sales.transferTo(salesPath);
        stock.transferTo(stockPath);

        try {
            LocalDate lastDate = validator.validateSales(salesPath);
            validator.validateStock(stockPath);

            return repository.save(new DatasetRecord(
                    salesPath.toString(),
                    stockPath.toString(),
                    lastDate
            ));
        } catch (IOException | RuntimeException e) {
            Files.deleteIfExists(salesPath);
            Files.deleteIfExists(stockPath);
            Files.deleteIfExists(directory);
            throw e;
        }
    }

    public DatasetRecord getLatestDataset() {
        return repository.findTopByOrderByCreatedAtDesc()
                .orElseThrow(() -> new ResponseStatusException(
                        HttpStatus.NOT_FOUND,
                        "Сначала загрузите файлы через POST /predict"
                ));
    }

    public void delete(DatasetRecord dataset) {
        repository.delete(dataset);
        Path salesPath = getSalesPath(dataset);
        Path stockPath = getStockPath(dataset);
        try {
            Files.deleteIfExists(salesPath);
            Files.deleteIfExists(stockPath);
            Files.deleteIfExists(salesPath.getParent());
        } catch (IOException ignored) {
            // Database rollback is the functional requirement; orphan cleanup
            // can be retried operationally if the filesystem is temporarily busy.
        }
    }

    public Path getSalesPath(DatasetRecord dataset) {
        return Path.of(dataset.getSalesFilePath());
    }

    public Path getStockPath(DatasetRecord dataset) {
        return Path.of(dataset.getStockFilePath());
    }
}
