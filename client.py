from random import random
import torch
from typing import Dict
import flwr as fl
import argparse
import warnings
import logging
import socket
import os
from absl import logging as absl_logging
import logging.handlers
import time
from prometheus_client import start_http_server, Histogram, Gauge

TRAINING_TIME = Histogram(
    'client_training_time_seconds', 
    'Time spent training a client',
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, float("inf")]
)

DATA_LOADING_TIME = Histogram(
    "client_data_loading_time_seconds",
    "Time spent preparing dataloaders for this client",
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, float("inf")]
)

DATA_SAMPLES = Gauge(
    "client_data_samples_total",
    "Number of training samples available to this client",
    labelnames=["client_id"],
)

EVAL_ACCURACY = Gauge("client_eval_accuracy",
                      "Last local evaluation accuracy", labelnames=["client_id"])
EVAL_LOSS = Gauge("client_eval_loss",
                  "Last local evaluation loss", labelnames=["client_id"])

DOWNLOAD_TIME = Histogram(
    "client_download_time_seconds",
    "Time spent downloading global parameters from the server",
    buckets=[0.05, 0.1, 0.25, 0.5, 1, 2, 5, float("inf")],
    labelnames=["client_id"],
)

UPLOAD_TIME = Histogram(
    "client_upload_time_seconds",
    "Time spent uploading updated parameters back to the server",
    buckets=[0.05, 0.1, 0.25, 0.5, 1, 2, 5, float("inf")],
    labelnames=["client_id"],
)

BATTERY_LEVEL = Gauge(
    "client_battery_level_percent",
    "Simulated battery percentage for this client",
    labelnames=["client_id"],
)

IS_CHARGING = Gauge(
    "client_is_charging",
    "Simulated charging state (1 charging, 0 not)",
    labelnames=["client_id"],
)

# imports model and data loading functions from task.py
from task import Net, load_data, train, test

def suppress_warnings():
    # Suppress all warnings
    warnings.filterwarnings("ignore")

    # Suppress Flower warnings
    logging.getLogger("flwr").setLevel(logging.ERROR)

    # Suppress gRPC warnings
    os.environ["GRPC_VERBOSITY"] = "ERROR"
    os.environ["GRPC_TRACE"] = "none"
    logging.getLogger("grpc").setLevel(logging.ERROR)

    # Suppress absl logging
    absl_logging.set_verbosity(absl_logging.ERROR)
    absl_logging.get_absl_handler().setLevel(logging.ERROR)

    # Suppress fork warnings
    logging.getLogger("multiprocessing").setLevel(logging.ERROR)

    # Remove all handlers associated with the root logger object
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Suppress deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Suppress torch/MNIST dataset warnings
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("torchvision").setLevel(logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser(description='Flower client for MNIST training')
    parser.add_argument('--server-address', type=str, default='server:8080',
                      help='Server address (default: server:8080)')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for training (default: 32)')
    parser.add_argument('--local-epochs', type=int, default=1,
                      help='Number of local epochs to train (default: 1)')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                      help='Learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9,
                      help='SGD momentum (default: 0.9)')
    parser.add_argument('--num-clients', type=int, default=int(os.getenv('NUM_CLIENTS', '3')),
                      help='Total number of clients (default: 3)')
    return parser.parse_args()



# Flower client
class MNISTClient(fl.client.NumPyClient):
    def __init__(self, args):
        self.model = Net()

        ## TODO: Improve partition ID assignment logic
        # Determine partition ID based on container IP address
        time.sleep(2)
        container_ip = socket.gethostbyname(socket.gethostname())
        
        # Extract the last octet of IP address
        ip_parts = container_ip.split('.')
        last_octet = int(ip_parts[-1])
        partition_id = last_octet - 2

        # Ensure partition ID is within valid range
        partition_id = partition_id % args.num_clients
        print(f"Final partition ID: {partition_id}")

        # Use federated data loading with partitioning
        start = time.time()
        self.trainloader, self.testloader = load_data(
            partition_id=partition_id,
            num_partitions=args.num_clients
        )
        DATA_LOADING_TIME.observe(time.time() - start)

        DATA_SAMPLES.labels(client_id=str(partition_id)).set(
            len(self.trainloader.dataset))
        

        self.args = args
        # Determine device
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # Move model to device once

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config: Dict[str, str]):
        print("\n" + "="*50)
        print("[FL] Starting local training round")

        self.battery_level = getattr(self, "battery_level", 100)
        self.battery_level = max(self.battery_level - random.uniform(5, 15), 0)
        BATTERY_LEVEL.labels(client_id=str(
            self.partition_id)).set(self.battery_level)
        
        if self.battery_level < 20:
            self.is_charging = 1
            self.battery_level = min(self.battery_level + random.uniform(10, 20), 100)

        else:
            self.is_charging = 0
        IS_CHARGING.labels(client_id=str(self.partition_id)).set(self.is_charging)

        # Measure training time with Prometheus histogram
        start_time = time.time()
        download_start = time.time()
        self.set_parameters(parameters)
        DOWNLOAD_TIME.labels(client_id=str(self.partition_id)).observe(
            time.time() - download_start)
        # Use local_epochs from command line args if not specified in server config
        epochs = int(config.get("local_epochs", self.args.local_epochs))

        train(self.model, self.trainloader, epochs=epochs, device=self.device, lr=self.args.learning_rate, momentum=self.args.momentum)
        
        print("[FL] Completed local training")
        print("="*50)

        # Record training time
        duration = time.time() - start_time
        TRAINING_TIME.observe(duration)
        print(f"[Metrics] Training round took {duration:.2f} seconds")

        upload_start = time.time()
        updated_parameters = self.get_parameters(config={})
        UPLOAD_TIME.labels(client_id=str(self.partition_id)).observe(
            time.time() - upload_start)
        
        return updated_parameters, len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        print("\n" + "="*50)
        print("[FL] Starting model evaluation")
        download_start = time.time()
        self.set_parameters(parameters)
        DOWNLOAD_TIME.labels(client_id=str(self.partition_id)).observe(
            time.time() - download_start)

        # Use the new test function to get both loss and accuracy
        loss, accuracy = test(self.model, self.testloader, device=self.device)
        EVAL_ACCURACY.labels(client_id=str(self.partition_id)).set(accuracy)
        EVAL_LOSS.labels(client_id=str(self.partition_id)).set(loss)

        print("[FL] Completed evaluation")
        print("="*50)
        # Return loss, number of examples, and a metrics dictionary
        return float(loss), len(self.testloader.dataset), {"accuracy": accuracy, "loss": loss}


if __name__ == "__main__":
    # Suppress warnings
    suppress_warnings()
    
    # Parse command line arguments
    args = parse_args()
    
    # Print client configuration
    print("\n" + "="*50)
    print("Flower Client Configuration:")
    print(f"Server Address: {args.server_address}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Local Epochs: {args.local_epochs}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Momentum: {args.momentum}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("="*50 + "\n")

    # Start Prometheus metrics server
    print(f"Starting Prometheus metrics server on port 8000...")
    start_http_server(8000)
    
    # Start Flower client
    print("Starting Flower client...")
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=MNISTClient(args),
    )
