"""
File to define models and model related functions for federated learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
import logging
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
import os
from datasets import disable_progress_bar

disable_progress_bar()

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
        print("Initializing FederatedDataset...")
        train_partitioner = IidPartitioner(num_partitions=num_partitions)
        test_partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="ylecun/mnist",
            partitioners={"train": train_partitioner, "test": test_partitioner},
        )
        print(f"FederatedDataset initialized with partition {partition_id}.")
    else:
        print(f"FederatedDataset already initialized using partition {partition_id}.")
    
    train_split = fds.load_partition(partition_id, "train")
    test_split = fds.load_partition(partition_id, "test")
    
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

    # Apply transforms to both train and test splits
    train_split = train_split.with_transform(apply_transforms)
    test_split = test_split.with_transform(apply_transforms)

    print(f"Partition {partition_id}: Train size = {len(train_split)}, Test size = {len(test_split)}")

    
    # Create DataLoaders
    trainloader = DataLoader(
        train_split, 
        batch_size=32, 
        shuffle=True,
        collate_fn=collate_fn
    )
    testloader = DataLoader(
        test_split, 
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