package hack.sber.service;

import java.io.IOException;
import java.io.Writer;
import java.nio.file.Path;

public interface PredictionEngine {

    void predict(
            Path salesFile,
            Path stockFile,
            String article,
            String warehouse,
            Writer output
    ) throws IOException;

    void predictAll(
            Path salesFile,
            Path stockFile,
            Writer output
    ) throws IOException;
}