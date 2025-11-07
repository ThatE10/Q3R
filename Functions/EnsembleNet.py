import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define the EnsNet architecture



# Helper function to verify dimensions
def check_dimensions():
    x = torch.randn(1, 1, 28, 28)  # Example input for MNIST
    model = EnsNet()

    # Check the base CNN output dimensions
    features = model.base_cnn(x)
    print(f"Feature maps shape: {features.shape}")

    # Verify linear layer input dimensions
    batch_size = features.size(0)
    flattened = features.reshape(batch_size, -1)
    print(f"Flattened shape: {flattened.shape}")

    # Check if the dimensions match the linear layer
    print(f"First linear layer expects: {model.feature_dim}")

    # Verify subnetwork dimensions
    channels_per_subnet = 2000 // 10
    for i in range(10):
        start_ch = i * channels_per_subnet
        end_ch = (i + 1) * channels_per_subnet
        subnet_features = features[:, start_ch:end_ch, :, :]
        flattened_subnet = subnet_features.reshape(batch_size, -1)
        print(f"Subnetwork {i} input shape: {flattened_subnet.shape}")


# Data augmentation transforms
train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, scale=(0.8, 1.2), translate=(0.08, 0.08), shear=0.3),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)



# Training function for alternating training of base CNN and subnetworks
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Step 1: Train base CNN
        optimizer.zero_grad()
        base_out, _ = model(data)
        loss = criterion(base_out, target)
        loss.backward()
        optimizer.step()

        # Step 2: Train subnetworks with fixed base CNN
        with torch.no_grad():
            features = model.base_cnn(data)

        for i in range(model.num_subnetworks):
            optimizer.zero_grad()

            # Get subset of feature maps for this subnetwork
            start_ch = i * model.channels_per_subnet
            end_ch = (i + 1) * model.channels_per_subnet
            subnet_features = features[:, start_ch:end_ch, :, :].detach().clone().requires_grad_(True)

            # Get prediction from subnetwork
            subnet_out = model.subnetworks[i](subnet_features.reshape(data.size(0), -1))
            loss = criterion(subnet_out, target)
            loss.backward()

            # Only update the parameters of the current subnetwork
            for param in model.subnetworks[i].parameters():
                if param.grad is not None:
                    param.data -= optimizer.param_groups[0]['lr'] * param.grad

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


# Testing function
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Get predictions using majority voting
            preds = model.predict(data)
            correct += (preds == target).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)
    error_rate = 100 - accuracy

    print(f'\nTest set: Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    print(f'Error rate: {error_rate:.2f}%')

    return error_rate


# Training loop
num_epochs = 1300  # As specified in the paper
for epoch in range(1, num_epochs + 1):
    train(model, device, train_loader, optimizer, criterion, epoch)

    # Test every 100 epochs to save time
    if epoch % 100 == 0:
        error_rate = test(model, device, test_loader)

        # Save model if it achieves a good error rate
        if error_rate < 0.3:  # arbitrary threshold
            torch.save(model.state_dict(), f'ensnet_mnist_epoch_{epoch}_error_{error_rate:.2f}.pt')

# Final evaluation
final_error_rate = test(model, device, test_loader)
print(f'Final error rate: {final_error_rate:.2f}%')