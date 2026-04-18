# =============================================================================
# React Dashboard Dockerfile — SITARAM
# =============================================================================

FROM node:20-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM node:20-alpine

WORKDIR /app

RUN npm install -g serve && \
    apk add --no-cache wget

COPY --from=builder /app/build ./build

EXPOSE 3000

# Listen on all interfaces
CMD ["serve", "-s", "build", "-l", "3000", "--listen", "tcp://0.0.0.0:3000"]

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=5 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:3000 || exit 1