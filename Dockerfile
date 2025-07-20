# Build stage
FROM python:3.11-slim AS builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=0 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Install system dependencies and Poetry
RUN pip install poetry

# Set working directory
WORKDIR /app

# Copy Poetry configuration files
COPY lol_pick_ml/pyproject.toml lol_pick_ml/poetry.lock* ./

# Configure Poetry and install dependencies
RUN poetry config virtualenvs.create false \
    && poetry config installer.max-workers 1 \
    && poetry config installer.parallel false \
    && export PIP_DEFAULT_TIMEOUT=600 \
    && export PIP_NO_BUILD_ISOLATION=1 \
    && export MAKEFLAGS="-j1" \
    && poetry install --only=main \
    && rm -rf $POETRY_CACHE_DIR \
    && pip cache purge \
    && rm -rf /root/.cache/pip \
    && find /usr/local/lib/python3.11/site-packages -name "*.pyc" -delete \
    && find /usr/local/lib/python3.11/site-packages -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Production stage
FROM python:3.11-slim AS production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/usr/local/bin:$PATH" \
    PYTHONPATH="/usr/local/lib/python3.11/site-packages"

# Create app user
RUN groupadd -g 1001 -r appgroup && \
    useradd -r -u 1001 -g appgroup appuser

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

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
