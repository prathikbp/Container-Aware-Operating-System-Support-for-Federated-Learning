"""
File to define models and model related functions for federated learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose
import logging
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

# Initialize FederatedDataset
fds = None

# Neural Network Model Definition
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
        return x

# Function to load data
# the data is loaded into partitions for federated learning
def load_data(partition_id: int, num_partitions: int):
    """Load partition MNIST data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="ylecun/mnist",
            partitioners={"train": partitioner},
        )
        print(f"FederatedDataset initialized for partition {partition_id}.")
    
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    
    # Define transforms
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.1307,), (0.3081,))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        # Apply transforms
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch
    
    # Custom collate function for DataLoader
    def collate_fn(batch):
        """Convert dict format to (data, target) tuple format for PyTorch."""
        images = torch.stack([item["image"] for item in batch])
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        return images, labels

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    
    # Create DataLoaders
    trainloader = DataLoader(
        partition_train_test["train"], 
        batch_size=32, 
        shuffle=True,
        collate_fn=collate_fn
    )
    testloader = DataLoader(
        partition_train_test["test"], 
        batch_size=32,
        collate_fn=collate_fn
    )
    return trainloader, testloader

def train(model, trainloader, epochs: int, device: torch.device, lr: float = 0.01, momentum: float = 0.9):
    """Train the model on the training set."""
    model.to(device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

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