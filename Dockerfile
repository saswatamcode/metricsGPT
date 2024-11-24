FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY Makefile requirements.txt ./

COPY ui/build/ ./ui/build/

COPY . .

RUN make deps

EXPOSE 8081

ENTRYPOINT ["python", "metricsGPT.py"]

CMD ["--server", "--config", "config.yaml"]
