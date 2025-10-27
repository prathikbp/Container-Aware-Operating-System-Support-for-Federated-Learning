import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose
from typing import Dict
import flwr as fl
import argparse
import warnings
import logging
import os
from absl import logging as absl_logging
import logging.handlers

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
    return parser.parse_args()

# Define the neural network model


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        # --- IMPROVEMENT 1 ---
        # Return raw logits instead of log_softmax
        # nn.CrossEntropyLoss (used in train) combines log_softmax and nll_loss
        return x

# Load data


def load_data(batch_size: int = 32):
    # Suppress MNIST download messages
    logging.getLogger("torchvision.datasets.mnist").setLevel(logging.WARNING)
    
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,))
    ])
    trainset = MNIST("./data", train=True, download=True, transform=transform)
    testset = MNIST("./data", train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size)
    return trainloader, testloader

# Train the model


def train(model, trainloader, epochs: int, device: torch.device):
    """Train the model on the training set."""
    model.to(device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    print(f"\n[Train] Starting training for {epochs} epochs on {device}")
    for epoch in range(epochs):
        running_loss = 0.0
        batch_count = 0
        for data, target in trainloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch_count += 1
        
        avg_loss = running_loss / batch_count
        print(f"[Train] Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

# Evaluate the model


def test(model, testloader, device: torch.device):
    """Evaluate the model on the test set."""
    print(f"\n[Eval] Starting evaluation on {device}")
    model.to(device)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, test_loss = 0, 0, 0.0

    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            # --- IMPROVEMENT 2 ---
            # Calculate loss as well as accuracy
            loss = criterion(outputs, target)
            test_loss += loss.item() * data.size(0)  # Accumulate weighted loss

            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    # Calculate average loss and accuracy
    avg_loss = test_loss / total
    accuracy = correct / total
    print(f"[Eval] Results - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy

# Flower client


class MNISTClient(fl.client.NumPyClient):
    def __init__(self, args):
        self.model = Net()
        self.trainloader, self.testloader = load_data(batch_size=args.batch_size)
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
        self.set_parameters(parameters)

        # Use local_epochs from command line args if not specified in server config
        epochs = int(config.get("local_epochs", self.args.local_epochs))

        train(self.model, self.trainloader, epochs=epochs, device=self.device)
        
        print("[FL] Completed local training")
        print("="*50)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        print("\n" + "="*50)
        print("[FL] Starting model evaluation")
        self.set_parameters(parameters)

        # Use the new test function to get both loss and accuracy
        loss, accuracy = test(self.model, self.testloader, device=self.device)

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
    
    print("Starting Flower client...")
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=MNISTClient(args),
    )
