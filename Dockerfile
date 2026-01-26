FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (needed for healthcheck)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (CPU-only PyTorch)
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy project files
COPY . .

# Expose API port (for documentation & tooling)
EXPOSE ${API_PORT}

# Start API using environment variable
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${API_PORT}"]
