# Federated Learning with Docker

This project implements a containerized Federated Learning system using Flower (flwr) and PyTorch, training on the MNIST dataset.

## Project Structure
```
.
├── Dockerfile          # Container configuration for both server and clients
├── docker-compose.yml # Docker services configuration
├── requirements.txt   # Python dependencies
├── server.py         # FL server implementation
├── client.py         # FL client implementation
└── data/            # MNIST dataset directory (auto-downloaded)
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

## Dependencies
- flwr==1.22.0
- torch==2.9.0
- torchvision==0.24.0
- numpy>=2.3.3

## Model Architecture
- Convolutional Neural Network (CNN)
- 2 convolutional layers
- 2 fully connected layers
- Dropout for regularization
- Log softmax output for digit classification

## How to Run

1. Build and start the system:
```bash
docker-compose up --build
```

2. Scale to more clients:
```bash
docker-compose up --scale client=3  # for 3 clients
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
- Adding Prometheus metrics for monitoring
- Setting up Grafana dashboards
- Implementing resource constraints for clients
- Adding performance visualization