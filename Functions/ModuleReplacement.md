# ModuleModificationHandler Usage Guide

This document provides an example usage guide for `ModuleModificationHandler.py`, which enables module replacements, adding LoRA adaptations, and applying weight regularization to neural network modules.

## Installation and Setup
Ensure you have `torch` installed:
```bash
pip install torch
```

Import necessary modules:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from ModuleModificationHandler import ModuleModificationHandler
```

## Example 1: Replacing a Module
This example demonstrates replacing `fc1` in a simple model with a custom `NewLinear` module.

```python
# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Custom linear module
class NewLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x) * 2  # Example modification

# Instantiate model
model = SimpleModel()
handler = ModuleModificationHandler(model)

# Replace fc1 with NewLinear
handler.replace_module("fc1", NewLinear(10, 5))

# Test
x = torch.randn(1, 10)
y = model(x)
print(y)
```

## Example 2: Adding L2 Weight Regularization
This example adds an L2 regularization hook to `fc2`.

```python
def l2_regularization(module):
    return 0.01 * torch.norm(module.weight, p=2)

handler.add_weight_regularizer("fc2", l2_regularization)
```

# Why Separate QKV Functions?

In Transformer-based models, Q, K, and V matrices are often stored as a single weight matrix, requiring special processing to extract and modify individual components. Functions like add_qkv_regularizer account for this structure by correctly partitioning the weight matrix before applying regularization.

## Example 3: Adding LoRA to QKV Layers
If using transformer-based architectures, LoRA can be added to the QKV module:

```python
handler.add_qkv_lora('blocks.0.attn.qkv', rank=4)
handler.add_qkv_regularizer('blocks.0.attn.qkv', regularization_type='l2')
```

## Example 4: Removing a Weight Regularizer
To remove a previously added weight regularizer:

```python
handler.remove_weight_regularizer("fc2")
```

## Summary
- `replace_module(module_name, replacement_module)`: Replaces a module.
- `add_weight_regularizer(module_name, regularization_fn)`: Adds a weight regularizer.
- `add_qkv_lora(module_name, rank)`: Adds LoRA adaptation.
- `add_qkv_regularizer(module_name, regularization_type)`: Adds QKV-specific regularization.
- `remove_weight_regularizer(module_name)`: Removes an added weight regularizer.

This guide provides a structured way to modify PyTorch models dynamically with `ModuleModificationHandler`.

