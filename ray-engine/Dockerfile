# =============================================================================
# Ray Engine Dockerfile — SITARAM
# =============================================================================
FROM rayproject/ray:2.8.0-py311

WORKDIR /app

USER root
RUN pip install --no-cache-dir kafka-python redis psycopg2-binary pandas numpy scipy sortedcontainers



EXPOSE 6380 8265
