package hack.sber.service;

import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Component;
import org.springframework.web.server.ResponseStatusException;

import java.io.IOException;
import java.io.Writer;
import java.nio.file.Path;

@Component
public class UnconfiguredPredictionEngine
        implements PredictionEngine {

    @Override
    public void predict(
            Path salesFile,
            Path stockFile,
            String article,
            String warehouse,
            Writer output
    ) throws IOException {
        throw new ResponseStatusException(
                HttpStatus.NOT_IMPLEMENTED,
                "ML-модель прогнозирования ещё не подключена"
        );
    }

    @Override
    public void predictAll(
            Path salesFile,
            Path stockFile,
            Writer output
    ) throws IOException {
        throw new ResponseStatusException(
                HttpStatus.NOT_IMPLEMENTED,
                "ML-модель прогнозирования ещё не подключена"
        );
    }
}