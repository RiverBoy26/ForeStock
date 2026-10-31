FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY models ./models
COPY training/history_snapshot.npz ./training/history_snapshot.npz
COPY training/final_raw_components.npz ./training/final_raw_components.npz
COPY training/keys.csv ./training/keys.csv
COPY forecasts ./forecasts
COPY docs/model_schema.json ./docs/model_schema.json
EXPOSE 8080
CMD ["uvicorn","app.api:api","--host","0.0.0.0","--port","8080"]
