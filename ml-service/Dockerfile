FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
WORKDIR /opt/nonton
RUN useradd --uid 10001 --create-home appuser
COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY models ./models
COPY forecasts ./forecasts
COPY snapshots ./snapshots
COPY model_manifest.json ./model_manifest.json
# Frozen artifacts remain root-owned and are only readable by the runtime user.
USER appuser
EXPOSE 8080
CMD ["python","-m","uvicorn","app.api:app","--host","0.0.0.0","--port","8080","--workers","1"]
