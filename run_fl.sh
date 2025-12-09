#!/bin/bash

# Federated Learning Deployment Script
# Usage:
#   ./run_fl.sh                          # uses values from fl_config.json
#   ./run_fl.sh 5                        # override client count
#   ./run_fl.sh --config custom.json     # custom config file
#   ./run_fl.sh --num-clients 4          # explicit override

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG_FILE="${SCRIPT_DIR}/fl_config.json"
CONFIG_FILE="$DEFAULT_CONFIG_FILE"
NUM_CLIENTS_OVERRIDE=""

usage() {
    cat <<'EOF'
Usage: ./run_fl.sh [options] [num_clients]

Options:
  -c, --config <path>     Path to JSON config file (default: fl_config.json)
      --num-clients <n>   Override client count without editing the config
  -h, --help              Show this message

Positional argument:
  num_clients             Same as --num-clients for backwards compatibility
EOF
}

POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config)
            CONFIG_FILE="${2:?Missing path after --config}"
            shift 2
            ;;
        --num-clients)
            NUM_CLIENTS_OVERRIDE="${2:?Missing number after --num-clients}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$NUM_CLIENTS_OVERRIDE" && ${#POSITIONAL[@]} -gt 0 ]]; then
    NUM_CLIENTS_OVERRIDE="${POSITIONAL[0]}"
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Config file '$CONFIG_FILE' not found." >&2
    exit 1
fi

# Load configuration into environment variables by shell-evaluating Python output.
eval "$(
python - <<'PY' "$CONFIG_FILE" "${NUM_CLIENTS_OVERRIDE:-}"
import json, shlex, sys

config_path = sys.argv[1]
override_clients = sys.argv[2] if len(sys.argv) > 2 else ""

with open(config_path, "r", encoding="utf-8") as fh:
    data = json.load(fh)

def emit(key: str, value):
    if value is None or key is None:
        return
    value_str = str(value)
    print(f"export {key}={shlex.quote(value_str)}")

server = data.get("server", {})
client = data.get("client", {})

emit("SERVER_ADDRESS", server.get("address", "0.0.0.0:8080"))
emit("SERVER_NUM_ROUNDS", server.get("num_rounds", 5))
emit("SERVER_FRACTION_FIT", server.get("fraction_fit", 1.0))
emit("SERVER_FRACTION_EVALUATE", server.get("fraction_evaluate", 1.0))
emit("SERVER_MIN_FIT_CLIENTS", server.get("min_fit_clients", 2))
emit("SERVER_MIN_EVALUATE_CLIENTS", server.get("min_evaluate_clients", 2))
emit("SERVER_MIN_AVAILABLE_CLIENTS", server.get("min_available_clients", 2))

emit("CLIENT_SERVER_ADDRESS", client.get("server_address", "server:8080"))
emit("CLIENT_BATCH_SIZE", client.get("batch_size", 32))
emit("CLIENT_LOCAL_EPOCHS", client.get("local_epochs", 1))
emit("CLIENT_LEARNING_RATE", client.get("learning_rate", 0.01))
emit("CLIENT_MOMENTUM", client.get("momentum", 0.9))

num_clients = override_clients or data.get("num_clients", 3)
emit("NUM_CLIENTS", num_clients)

print(f'export CONFIG_FILE={shlex.quote(config_path)}')
PY
)"

echo "=================================================="
echo "Starting Federated Learning with $NUM_CLIENTS clients"
echo "Config file: $CONFIG_FILE"
echo "Server address: $SERVER_ADDRESS (rounds: $SERVER_NUM_ROUNDS)"
echo "Client params: batch=$CLIENT_BATCH_SIZE, epochs=$CLIENT_LOCAL_EPOCHS, lr=$CLIENT_LEARNING_RATE"
echo "=================================================="

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

# Start remaining services (clients + optional stack)
MONITORING_SERVICES=("prometheus" "grafana")
OPTIONAL_SERVICES=("manager")

services_to_start=("client" "${OPTIONAL_SERVICES[@]}" "${MONITORING_SERVICES[@]}")
echo "Starting $NUM_CLIENTS FL clients plus services: ${services_to_start[*]}..."
docker-compose up -d --scale client=$NUM_CLIENTS "${services_to_start[@]}"

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
