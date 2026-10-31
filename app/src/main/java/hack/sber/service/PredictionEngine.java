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
            String variant,
            int startWeek,
            int endWeek,
            Writer output
    ) throws IOException;

    void predictAll(
            Path salesFile,
            Path stockFile,
            String variant,
            int startWeek,
            int endWeek,
            Writer output
    ) throws IOException;
}
