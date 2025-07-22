# Build stage
FROM python:3.11-slim AS builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN pip install poetry

# Set working directory
WORKDIR /app

# Copy Poetry configuration files
COPY lol_pick_ml/pyproject.toml lol_pick_ml/poetry.lock* ./

# Export dependencies to requirements.txt and install to a custom directory
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes && \
    pip install --prefix=/install --no-warn-script-location -r requirements.txt

# Production stage
FROM python:3.11-slim AS production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/usr/local/lib/python3.11/site-packages"

# Create app user
RUN groupadd -g 1001 -r appgroup && \
    useradd -r -u 1001 -g appgroup appuser

# Set working directory
WORKDIR /app

# Copy the installed packages from builder stage
COPY --from=builder /install /usr/local

# Verify that uvicorn is available before copying application code
RUN python -c "import uvicorn; print('uvicorn successfully imported')" || (echo "uvicorn import failed" && exit 1)

# Copy application code
COPY --chown=appuser:appgroup lol_pick_ml/ .

# Create data directory structure (actual data should be mounted as volume)
RUN mkdir -p data && chown -R appuser:appgroup data

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Start the application
CMD ["python", "main.py"]
