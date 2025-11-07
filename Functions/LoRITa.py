from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


class LoRITaLinear(nn.Module):
    def __init__(self, in_features, out_features, N=4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.N = N

        if N == 1:
            self.net1 = nn.Sequential(nn.Linear(in_features, out_features))
        else:
            layers = OrderedDict()
            for i in range(N - 1):
                layers[f'compress_{i}'] = nn.Linear(in_features, in_features, bias=False)
            layers[f'compress_{N - 1}'] = nn.Linear(in_features, out_features)
            self.net1 = nn.Sequential(layers)

    def forward(self, x):
        x = self.net1(x)
        return x

    def get_layer_reference(self):
        weight_refs = []
        for name, module in self.net1.named_modules():
            if isinstance(module, nn.Linear):
                weight_refs.append(module)
        return weight_refs

    def get_combined_weight_factors(self):
        weights = []
        for name, module in self.net1.named_modules():
            if isinstance(module, nn.Linear):
                weights.append(module.weight.data)

        if not weights:
            return None  # or raise ValueError("No linear layers found.")

        # Perform matrix multiplication in forward order
        combined = weights[0]
        for weight in weights[1:]:
            combined = weight @ combined
        return combined


    def __repr__(self):
        # Return a string representation of the instance with the parameters (in_features, out_features, rank)
        return f"LoRITa Module (in_features={self.in_features}, out_features={self.out_features}, N={self.N})"


class LoRITaQKV(nn.Module):
    def __init__(self, original_qkv, N=3):
        super().__init__()

        # Get original weight dimensions
        total_dim = original_qkv.weight.shape[0]
        dim = total_dim // 3  # Assuming Q, K, V are concatenated

        # LoRA layers for Q, K, V
        self.q_lorita = LoRITaLinear(original_qkv.in_features, dim, N=N)
        self.k_lorita = LoRITaLinear(original_qkv.in_features, dim, N=N)
        self.v_lorita = LoRITaLinear(original_qkv.in_features, dim, N=N)

    def forward(self, x):
        q = self.q_lorita(x)
        k = self.k_lorita(x)
        v = self.v_lorita(x)

        return torch.cat([q, k, v], dim=-1)

    def get_layer_reference(self):
        # Collect weights from all three LoRITaLinear components
        q_weights = self.q_lorita.get_layer_reference()
        k_weights = self.k_lorita.get_layer_reference()
        v_weights = self.v_lorita.get_layer_reference()

        # Combine all weights
        all_weights = q_weights + k_weights + v_weights

        return all_weights

    def get_combined_weight_factors(self):
        # Collect weights from all three LoRITaLinear components
        q_weights = self.q_lorita.get_combined_weight_factors()
        k_weights = self.k_lorita.get_combined_weight_factors()
        v_weights = self.v_lorita.get_combined_weight_factors()

        return [q_weights, k_weights, v_weights]

""" Source code from: https://github.com/XitongSystem/LoRITa/blob/main/Compress_Code/arch/fcn.py

from torch import nn
from collections import OrderedDict

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, factor = 1):
        super().__init__()
        self.norm = nn.BatchNorm1d(hidden_dim)
        if factor == 1:
            self.net1 = nn.Linear(dim, hidden_dim)

        else:
            self.net1 = OrderedDict()
            for i in range(factor - 1):
                self.net1['compress_'+str(i)] = nn.Linear(dim, dim, bias=False)
            self.net1['compress_'+str(factor - 1)] = nn.Linear(dim, hidden_dim)
            self.net1 = nn.Sequential(self.net1)

    def forward(self, x):
        x = self.net1(x)
        x = self.norm(x)
        return x

class FCN(nn.Module):
    def __init__(self, *, image_size, num_classes, dim, depth, factor=1):
        super().__init__()

        self.layers = [FeedForward(image_size, dim, factor), nn.ReLU()]
        for i in range(depth - 2):
            self.layers.append(FeedForward(dim, dim, factor))
            self.layers.append(nn.ReLU())
        self.layers.append(FeedForward(dim, num_classes, factor))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, img):
        return self.layers(img.view(img.shape[0],-1))
"""
