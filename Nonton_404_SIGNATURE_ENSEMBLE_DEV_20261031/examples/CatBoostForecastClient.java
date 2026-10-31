import java.net.URI;
import java.net.http.*;
import java.time.Duration;

// Java 21 minimal example: GET /v1/forecasts?limit=5
public class CatBoostForecastClient {
  public static void main(String[] args) throws Exception {
    var client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(10)).build();
    var request = HttpRequest.newBuilder(URI.create("http://localhost:8080/v1/forecasts?limit=5"))
      .header("X-API-Key", System.getenv("NONTON_API_KEY"))
      .GET().build();
    var response = client.send(request, HttpResponse.BodyHandlers.ofString());
    System.out.println(response.statusCode());
    System.out.println(response.body());
  }
}
