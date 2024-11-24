FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY metricsGPT.py .
COPY config.yaml .
COPY LICENSE .
COPY ui/build ui/build

EXPOSE 8081

ENTRYPOINT ["python", "metricsGPT.py"]
