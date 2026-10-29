# ForeStock

ForeStock combines a Spring Boot business API with the immutable Nonton v7.4
LightGBM model service. New sales and stock observations rebuild rolling
features; the 41 model files, their categorical encodings, and model version are
never retrained or modified.

## Run

```powershell
Copy-Item .env.example .env
# Replace both secrets in .env, then:
docker compose up --build
```

- Spring API: `http://127.0.0.1:8000`
- ML Swagger: `http://127.0.0.1:8080/docs`

## Dynamic forecast API

Upload the latest observations and receive a CSV forecast for the next nine
complete weeks:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict?variant=selected" `
  -F "sales=@sales.csv" `
  -F "stock=@stock.csv" `
  -o forecast.csv
```

After an upload, the latest dataset is stored by the backend and can be reused:

```text
GET /predict/all?variant=selected
GET /predict?article=...&warehouse=...&variant=selected
```

Supported variants are `selected`, `conservative`, `smape_experimental`, and
`baseline`.

## Immutable-model guarantees and data contract

- The ML service validates SHA-256 for all 41 models and the reference snapshot
  during startup and refuses to become ready if an artifact changed. Its Docker
  filesystem is read-only and the artifacts remain owned by root.
- No training endpoint exists. Uploaded files are temporary and only produce
  feature matrices for inference.
- At least 52 weeks of sales history are required. The latest Sunday covered by
  both files becomes the cutoff; partial following weeks are ignored.
- Input CSV columns follow the original hackathon schema. Stock accepts either
  `article` or `char_article`.
- The model catalog is frozen to its original 11 warehouses and 799 articles.
  A subset is allowed, but any unknown warehouse or article is rejected with
  HTTP 422 because it has no valid trained categorical encoding.
- Forecasts always contain nine Monday week starts following the computed cutoff.

The model weights remain identical for every request. Forecast quality outside
the original training distribution is not guaranteed; dynamic inference is
extrapolation with the frozen v7.4 model, not retraining.
