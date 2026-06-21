FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy project specification files
COPY pyproject.toml uv.lock ./

# Install dependencies into system Python environment
RUN uv pip install --system --no-cache -r pyproject.toml

# Copy backend application and sample datasets
COPY app ./app
COPY data ./data

EXPOSE 8000

ENV PYTHONPATH=/app

# Run uvicorn on the port assigned dynamically by Railway, defaulting to 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
