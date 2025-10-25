# Use a lightweight Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    netcat-traditional \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create a data directory (used by clients for MNIST or other datasets)
RUN mkdir -p /app/data

# Prevent creation of .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Expose ports used by Flower server (8080) and Prometheus metrics (8000)
EXPOSE 8080 8000

# Default command (overridden in docker-compose.yml)
CMD ["python", "server.py"]
