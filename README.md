# Federated Learning with Docker

This project implements a containerized Federated Learning system using Flower (flwr) and PyTorch, training on the MNIST dataset.

## Project Structure
```
.
├── Dockerfile          # Container configuration for both server and clients
├── docker-compose.yml  # Docker services configuration
├── requirements.txt    # Python dependencies
├── server.py           # FL server implementation
├── client.py           # FL client implementation
├── task.py             # FL model and data loaders
├── run_fl.sh           # deployment script
└── data/               # MNIST dataset directory (auto-downloaded)
```

## Features
- Federated Learning implementation using Flower framework
- MNIST handwritten digit classification using PyTorch
- Containerized setup with Docker and Docker Compose
- Support for multiple FL clients running in parallel
- Shared dataset volume between containers
- Health checks to ensure proper service startup

## Technical Stack
- Python 3.11
- PyTorch for deep learning
- Flower (flwr) for federated learning
- Docker for containerization
- Docker Compose for service orchestration

## Requirements

### System Requirements
- Docker Engine 24.0 or later
- Docker Compose v2.0 or later
- At least 4GB RAM
- 10GB free disk space

### Python Dependencies
- flwr==1.22.0
- torch==2.9.0
- torchvision==0.24.0
- numpy>=2.3.3

### Operating System Support
- Linux (recommended)
- macOS
- Windows with WSL2

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/prathikbp/Container-Aware-Operating-System-Support-for-Federated-Learning.git
cd Container-Aware-Operating-System-Support-for-Federated-Learning
```

2. (Optional) Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/macOS
# or
.\venv\Scripts\activate  # On Windows
```

3. Install dependencies (if running without Docker):
```bash
pip install -r requirements.txt
```

## Running the Project
### Using the deployment script

Before running the deployment script, ensure executable permissions are set:

```bash
chmod +x run_fl.sh
```

```bash
# Run with 5 clients (default is 3)
./run_fl.sh 5
```

### Using Docker (Recommended)

1. Build and start the system with one server and one client:
```bash
docker compose up --build
```

2. Scale to multiple clients (e.g., 3 clients):
```bash
docker compose up --scale client=3
```

3. Monitor the training progress in the logs:
```bash
docker compose logs -f
```

4. Stop the system:
```bash
docker compose down
```

### Running Without Docker

1. Start the server:
```bash
# Basic server start
python server.py

# Start server with custom address and rounds
python server.py --server-address "0.0.0.0:8080" --num-rounds 10

# Start server with minimum client requirements
python server.py --min-fit-clients 3 --min-evaluate-clients 3 --min-available-clients 3

# Start server with custom client fractions
python server.py --fraction-fit 0.8 --fraction-evaluate 0.8
```

2. Start one or more clients (in separate terminals):
```bash
# Basic client start
python client.py

# Start client with custom server address
python client.py --server-address "localhost:8080"

# Start client with custom training parameters
python client.py --batch-size 64 --local-epochs 2 --learning-rate 0.001 --momentum 0.9
```

Note: To run multiple clients, open multiple terminal windows and run the client script with different client IDs.

Example workflow for 3 clients:
```bash
# Terminal 1 - Start Server
python server.py --num-rounds 10 --min-fit-clients 3 --min-evaluate-clients 3 --min-available-clients 3

# Terminal 2 - Start Client 1
python client.py --batch-size 32 --local-epochs 1 --learning-rate 0.01

# Terminal 3 - Start Client 2
python client.py --batch-size 32 --local-epochs 1 --learning-rate 0.01

# Terminal 4 - Start Client 3
python client.py --batch-size 32 --local-epochs 1 --learning-rate 0.01
```

### Troubleshooting
- If you see "address already in use" error, ensure no other FL server is running
- If clients can't connect, check if the server is properly started
- For CUDA errors, ensure PyTorch is properly installed with CUDA support

## Model Architecture
- Convolutional Neural Network (CNN)
- 2 convolutional layers
- 2 fully connected layers
- Dropout for regularization
- Log softmax output for digit classification

## How to Run

1. Build and start the system:
```bash
docker compose up --build
```

2. Scale to more clients:
```bash
docker compose up --scale client=3  # for 3 clients
```

## Federated Learning Configuration
- Training Rounds: 5
- Client Fraction: 1.0 (all clients participate)
- Minimum Clients: 2
- Aggregation Strategy: FedAvg (Federated Averaging)
- Metrics: Accuracy with weighted averaging

## Docker Configuration
- Base Image: python:3.11-slim
- Shared Volume: ./data for MNIST dataset
- Network: Automatic Docker network for service discovery
- Health Check: Server readiness check before client startup

## Next Steps
- Better way to name the partition for each clients
- Adding a config file instead of CLI parameter passing
- Adding Prometheus metrics for monitoring
- Setting up Grafana dashboards
- Implementing resource constraints for clients
- Adding performance visualization