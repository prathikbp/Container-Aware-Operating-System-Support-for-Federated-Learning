#!/bin/bash

# Federated Learning Deployment Script
# Usage: ./run_fl.sh [number_of_clients] [additional_args]

set -e

# Default number of clients
NUM_CLIENTS=${1:-2}
shift || true  # Remove first argument, keep the rest

echo "=================================================="
echo "Starting Federated Learning with $NUM_CLIENTS clients"
echo "=================================================="

# Export environment variable for docker-compose
export NUM_CLIENTS

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down FL system..."
    docker-compose down -v
}

# Set trap to cleanup on script exit
trap cleanup EXIT INT TERM

# Check if Docker and docker-compose are available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "Error: docker-compose is not installed or not in PATH"
    exit 1
fi

# Build the images first
echo "🏗️  Building Docker images..."
docker-compose build

# Start the server first
echo "Starting FL server..."
docker-compose up -d server

# Wait for server to be healthy
echo "Waiting for server to be ready..."
timeout=60
elapsed=0
while [ $elapsed -lt $timeout ]; do
    if docker-compose exec -T server python -c "import socket; s = socket.socket(); s.settimeout(1); result = s.connect_ex(('localhost', 8080)); s.close(); exit(0 if result == 0 else 1)" 2>/dev/null; then
        echo "Server is ready!"
        break
    fi
    sleep 2
    elapsed=$((elapsed + 2))
    echo "   Still waiting... (${elapsed}s/${timeout}s)"
done

if [ $elapsed -ge $timeout ]; then
    echo "Error: Server failed to start within ${timeout} seconds"
    exit 1
fi

# Start the clients with scaling
echo "Starting $NUM_CLIENTS FL clients..."
docker-compose up -d --scale client=$NUM_CLIENTS client

echo ""
echo "Federated Learning system is running!"
echo ""
echo "Monitor the logs:"
echo "   Server logs:  docker-compose logs -f server"
echo "   Client logs:  docker-compose logs -f client"
echo "   All logs:     docker-compose logs -f"
echo ""
echo "To stop the system: Ctrl+C or docker-compose down"
echo ""

# Follow logs
docker-compose logs -f