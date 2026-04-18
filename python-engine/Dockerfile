# =============================================================================
# Python Engine Dockerfile — SITARAM
# Runs: feature computation, order book reconstruction, quoting engine
# =============================================================================
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py /app/main.py
COPY src/ /app/src/
COPY config/ /app/config/
COPY tests/ /app/tests/

# Health check endpoint runs on port 8000
EXPOSE 8000

CMD ["python", "main.py"]
