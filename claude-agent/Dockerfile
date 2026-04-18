# =============================================================================
# Claude Agent Dockerfile
# =============================================================================
FROM python:3.11-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    anthropic \
    psycopg2-binary \
    redis \
    flask \
    flask-socketio \
    flask-cors \
    gevent \
    gevent-websocket

# Copy source
COPY src/ /app/src/
COPY prompts/ /app/prompts/

EXPOSE 8001

CMD ["python", "/app/src/agent.py"]
