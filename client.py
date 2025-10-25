import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose
from typing import Dict
import flwr as fl

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


def load_data():
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,))
    ])
    trainset = MNIST("./data", train=True, download=True, transform=transform)
    testset = MNIST("./data", train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32)
    return trainloader, testloader

# Train the model


def train(model, trainloader, epochs: int, device: torch.device):
    """Train the model on the training set."""
    model.to(device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for _ in range(epochs):
        for data, target in trainloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

# Evaluate the model


def test(model, testloader, device: torch.device):
    """Evaluate the model on the test set."""
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
    return avg_loss, accuracy

# Flower client


class MNISTClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = Net()
        self.trainloader, self.testloader = load_data()
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
        self.set_parameters(parameters)

        # --- IMPROVEMENT 4 ---
        # Read epochs from server config, default to 1 if not sent
        epochs = int(config.get("local_epochs", 1))

        train(self.model, self.trainloader, epochs=epochs, device=self.device)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        # --- IMPROVEMENT 3 ---
        # Use the new test function to get both loss and accuracy
        loss, accuracy = test(self.model, self.testloader, device=self.device)

        # Return loss, number of examples, and a metrics dictionary
        return float(loss), len(self.testloader.dataset), {"accuracy": accuracy, "loss": loss}


if __name__ == "__main__":
    # In Docker, 'server' hostname resolves to the server service
    # If running locally, you might need to change this to "0.0.0.0:8080"
    fl.client.start_numpy_client(
        server_address="server:8080",
        client=MNISTClient(),
    )
