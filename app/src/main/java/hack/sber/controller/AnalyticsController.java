package hack.sber.controller;

import hack.sber.model.ForecastAnalyticsResponse;
import hack.sber.service.ForecastAnalyticsService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/analytics")
public class AnalyticsController {

    private final ForecastAnalyticsService analyticsService;

    public AnalyticsController(ForecastAnalyticsService analyticsService) {
        this.analyticsService = analyticsService;
    }

    @GetMapping("/latest")
    public ForecastAnalyticsResponse latest() {
        return analyticsService.getLatest();
    }
}
