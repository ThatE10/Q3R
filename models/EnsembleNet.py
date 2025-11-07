import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class EnsNet(nn.Module):
    def __init__(self, num_subnetworks=10):
        super(EnsNet, self).__init__()

        # Base CNN part
        self.base_cnn = nn.Sequential(
            # Conv3-64 (zero padding)
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.35),

            # Conv3-128
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.35),

            # Conv3-256 (zero padding)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Size becomes 12x12
            nn.Dropout(0.35),

            # Conv3-512 (zero padding)
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.35),

            # Conv3-1024
            nn.Conv2d(512, 1024, kernel_size=3),  # Size becomes 10x10
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Dropout(0.35),

            # Conv3-2000 (zero padding)
            nn.Conv2d(1024, 2000, kernel_size=3, padding=1),
            nn.BatchNorm2d(2000),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Size becomes 5x5
            nn.Dropout(0.35)
        )

        # Calculate feature dimensions
        self.feature_dim = 2000 * 5 * 5

        # Fully connected part of the base CNN
        self.base_fc = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

        # Create subnetworks
        self.num_subnetworks = num_subnetworks
        self.channels_per_subnet = 2000 // num_subnetworks

        # Each subnetwork takes a subset of feature maps
        self.subnetworks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.channels_per_subnet * 5 * 5, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10)
            ) for _ in range(num_subnetworks)
        ])

    def forward(self, x):
        if not self.training:
            return self.predict(x)
        # Get feature maps from base CNN
        features = self.base_cnn(x)
        batch_size = features.size(0)

        # Base CNN prediction (fully connected part)
        base_out = self.base_fc(features.reshape(batch_size, -1))

        # Get predictions from subnetworks
        subnet_outputs = []
        for i in range(self.num_subnetworks):
            # Get subset of feature maps for this subnetwork
            start_ch = i * self.channels_per_subnet
            end_ch = (i + 1) * self.channels_per_subnet
            subnet_features = features[:, start_ch:end_ch, :, :]

            # Get prediction from subnetwork
            subnet_out = self.subnetworks[i](subnet_features.reshape(batch_size, -1))
            subnet_outputs.append(subnet_out)

        # Return base CNN output and all subnetwork outputs
        return base_out

    def predict(self, x):
            # Get feature maps from base CNN
        features = self.base_cnn(x)
        batch_size = features.size(0)

        # Base CNN prediction (fully connected part)
        base_out = self.base_fc(features.reshape(batch_size, -1))

        # Get predictions from subnetworks
        subnet_outputs = []
        for i in range(self.num_subnetworks):
            # Get subset of feature maps for this subnetwork
            start_ch = i * self.channels_per_subnet
            end_ch = (i + 1) * self.channels_per_subnet
            subnet_features = features[:, start_ch:end_ch, :, :]

            # Get prediction from subnetwork
            subnet_out = self.subnetworks[i](subnet_features.reshape(batch_size, -1))
            subnet_outputs.append(subnet_out)

        # Get predictions from base CNN
        base_pred = torch.argmax(base_out, dim=1)

        # Get predictions from subnetworks
        subnet_preds = [torch.argmax(out, dim=1) for out in subnet_outputs]

        # Stack all predictions
        all_preds = torch.stack([base_pred] + subnet_preds, dim=1)

        # Majority vote
        final_pred = torch.mode(all_preds, dim=1).values

        return final_pred

    def train_subnetworks(self, inputs, labels, optimizer, criterion):
        # Get feature maps with base CNN frozen
        with torch.no_grad():
            features = self.base_cnn(inputs)

        subnet_losses = []
        for i in range(self.num_subnetworks):
            optimizer.zero_grad()

            # Get subset of feature maps for this subnetwork
            start_ch = i * self.channels_per_subnet
            end_ch = (i + 1) * self.channels_per_subnet
            subnet_features = features[:, start_ch:end_ch, :, :].detach().clone().requires_grad_(True)

            # Get prediction from subnetwork
            subnet_out = self.subnetworks[i](subnet_features.reshape(inputs.size(0), -1))
            loss = criterion(subnet_out, labels)
            loss.backward()

            # Only update the parameters of the current subnetwork
            for param in self.subnetworks[i].parameters():
                if param.grad is not None:
                    param.data -= optimizer.param_groups[0]['lr'] * param.grad

            subnet_losses.append(loss.item())

        return sum(subnet_losses) / len(subnet_losses)  # Average loss across subnetworks