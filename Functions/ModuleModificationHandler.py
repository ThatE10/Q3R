from typing import Callable, Dict
import torch
from torch import nn
from torch import norm as t_norm
from torch import randn as t_randn
from torch import sum as t_sum
from torch import abs as t_abs

from Functions.LoRITa import LoRITaQKV, LoRITaLinear
from Functions.Lora import LoRAQKV, LoRALinear, TruncateLoRALinear, TruncateLoRAQKVModule


class ModuleReplacementHandler:
    def __init__(self, model: nn.Module):
        self.original_model = model
        self.module_replacements: Dict[str, nn.Module] = {}
        self.weight_hooks: Dict[str, Callable] = {}

    def _get_module_by_name(self, module_name: str) -> nn.Module:
        """
        Retrieve a module by its full name path

        Args:
            module_name (str): Dot-separated path to the module

        Returns:
            nn.Module: The requested module
        """
        name_parts = module_name.split('.')
        current = self.original_model

        for part in name_parts:
            current = getattr(current, part)

        return current

    import re

    def replace_module(self, module_name: str, replacement_module: nn.Module, module: nn.Module = None):
        """
        Replace a specific module in the model
        Args:
            module_name (str): Dot-separated path to the module
            replacement_module (nn.Module): New module to replace the existing one
            module (nn.Module, optional): Existing module to verify against the module name
        Returns:
            self: Allows method chaining
        """
        # Split the module path
        name_parts = module_name.split('.')

        # If a module is provided, verify it matches the module name
        if module is not None:
            # Get the module by name to verify
            verified_module = self._get_module_by_name(module_name)
            if module is not verified_module:
                raise ValueError(f"Provided module does not match the module at {module_name}")
        else:
            # If no module is provided, retrieve it by name
            module = self._get_module_by_name(module_name)

        # Navigate to the parent module
        parent = self.original_model
        for part in name_parts[:-1]:
            parent = getattr(parent, part)

        # Replace the module
        replacement_module.to(next(module.parameters()).device)

        setattr(parent, name_parts[-1], replacement_module)

        # Store the replacement
        self.module_replacements[module_name] = replacement_module

        return self

    def add_weight_regularizer(self, module_name: str, regularization_fn: Callable):
        """
        Add a weight regularization hook to a specific module

        Args:
            module_name (str): Dot-separated path to the module
            regularization_fn (Callable): Function to apply regularization
        """

        def hook(module, grad_input, grad_output):
            regularized_grad = regularization_fn(module)
            return (regularized_grad,)

        module = self._get_module_by_name(module_name)

        handle = module.register_full_backward_hook(hook)

        self.weight_hooks[module_name] = handle

        return self

    def add_qkv_regularizer(self, module_name: str, regularization_type='l2'):
        """
        Add regularization specifically to Q, K, V weights

        Args:
            module_name (str): Path to the QKV module
            regularization_type (str): Type of regularization
        """

        def qkv_regularization(module):
            # Find the original QKV weights
            total_dim = module.weight.shape[0]
            dim = total_dim // 3

            # Extract Q, K, V weights
            q_weight = module.weight[:dim]
            k_weight = module.weight[dim:2 * dim]
            v_weight = module.weight[2 * dim:]

            # Apply regularization
            if regularization_type == 'l2':
                return 0.01 * (t_norm(q_weight) +
                               t_norm(k_weight) +
                               t_norm(v_weight))
            elif regularization_type == 'l1':
                return 0.01 * (t_sum(t_abs(q_weight)) +
                               t_sum(t_abs(k_weight)) +
                               t_sum(t_abs(v_weight)))
            else:
                raise ValueError(f"Unsupported regularization type: {regularization_type}")

        # Add the regularization hook
        self.add_weight_regularizer(module_name, qkv_regularization)

        return self

    def add_qkv_lora(self, module_name: str, rank: int = 4):
        qkv_module = self._get_module_by_name(module_name)

        self.replace_module(module_name, LoRAQKV(qkv_module, rank))

        return self

    def add_lora_layer(self, module_name: str, rank: int = 4):
        """
        Add LoRA (Low-Rank Adaptation) to a linear layer

        Args:
            module_name (str): Dot-separated path to the linear module
            rank (int): Rank of the LoRA adaptation
        """
        # Find the original module
        original_module = self._get_module_by_name(module_name)

        # Check if the module is already a LoRA layer or wrapped

        if not isinstance(original_module, nn.Linear):
            raise ValueError(f"Module {module_name} must be a Linear layer")

        self.replace_module(module_name,
                            LoRALinear(original_module.in_features, original_module.out_features, rank=rank))

        return self

    def add_qkv_lorita(self, module_name: str, N: int = 3):
        qkv_module = self._get_module_by_name(module_name)

        self.replace_module(module_name, LoRITaQKV(qkv_module, N))

        return self

    def add_lorita_layer(self, module_name: str, N: int = 3):
        """
        Add LoRITa (Low-Rank Induced Training Through Linear Layers) to a linear layer

        Args:
            module_name (str): Dot-separated path to the linear module
            N (int): Number of layers used within the Lorita Module
        """
        # Find the original module
        original_module = self._get_module_by_name(module_name)

        # Check if the module is already a LoRITa layer or wrapped

        if not isinstance(original_module, nn.Linear):
            raise ValueError(f"Module {module_name} must be a Linear layer")

        self.replace_module(module_name,
                            LoRITaLinear(original_module.in_features, original_module.out_features, N=N))

        return self

    def simplify_lorita_module(self, module_name, lorita_module=None):
        """
        Simplify a LoRITaLinear module by replacing it with an equivalent nn.Linear module
        with weights computed through multiplication of all sublinear layers.

        Args:
            handler (ModuleReplacementHandler): The module replacement handler
            module_name (str): Name of the module to replace
            lorita_module (LoRITaLinear, optional): The module to replace, if None it will be retrieved by name

        Returns:
            ModuleReplacementHandler: The handler for method chaining
        """
        # Get the module if not provided
        if lorita_module is None:
            lorita_module = self._get_module_by_name(module_name)

        # Verify this is a LoRITaLinear module
        if not isinstance(lorita_module, LoRITaLinear):
            raise ValueError(f"Module {module_name} must be a LoRITaLinear layer")

        # Create a standard Linear replacement
        replacement = nn.Linear(lorita_module.in_features, lorita_module.out_features)

        # Compute equivalent weights
        with torch.no_grad():

            # Case with multiple layers - compute product of matrices
            weight = torch.eye(lorita_module.in_features, device=next(lorita_module.parameters()).device)

            # Multiply weights in sequence (all but the last layer)
            for i in range(lorita_module.N - 1):
                layer = lorita_module.net1[i]
                weight = layer.weight.data @ weight

            # Apply final layer
            final_layer = lorita_module.net1[lorita_module.N - 1]
            weight = final_layer.weight.data @ weight

            # Get bias from final layer
            bias = final_layer.bias.data if hasattr(final_layer, 'bias') and final_layer.bias is not None else None

            # Set the weights and bias in the replacement module
            replacement.weight.data = weight
            if bias is not None:
                replacement.bias.data = bias

        # Replace the module using the handler
        print(f"Replacing {module_name} with {replacement}")
        self.replace_module(module_name, replacement, lorita_module)

        return self

    def simplify_lorita_qkv_module(self, module_name, lorita_module: LoRITaQKV = None):
        """
        Simplify a LoRITaQKV module by replacing it with a standard nn.Linear QKV module
        in timm style (concatenated QKV weights).

        Args:
            handler (ModuleReplacementHandler): The module replacement handler
            module_name (str): Name of the module to replace
            lorita_module (LoRITaQKV, optional): The module to replace, if None it will be retrieved by name

        Returns:
            ModuleReplacementHandler: The handler for method chaining
        """
        # Get the module if not provided
        if lorita_module is None:
            lorita_module = self._get_module_by_name(module_name)

        # Verify this is a LoRITaQKV module
        if not isinstance(lorita_module, LoRITaQKV):
            raise ValueError(f"Module {module_name} must be a LoRITaQKV layer")

        # Get dimensions
        in_features = lorita_module.q_lorita.in_features
        dim = lorita_module.q_lorita.out_features

        # Create a standard Linear replacement (3 * dim for concatenated Q, K, V)
        replacement = nn.Linear(in_features, 3 * dim)

        # Process each component (q, k, v)
        components = ['q_lorita', 'k_lorita', 'v_lorita']

        with torch.no_grad():
            # Initialize bias if needed
            has_bias = any(hasattr(getattr(lorita_module, comp).net1[
                                       -1 if isinstance(getattr(lorita_module, comp).net1, nn.Sequential) else None],
                                   'bias') and
                           getattr(lorita_module, comp).net1[-1 if isinstance(getattr(lorita_module, comp).net1,
                                                                              nn.Sequential) else None].bias is not None
                           for comp in components)

            if has_bias:
                replacement.bias = nn.Parameter(torch.zeros(3 * dim, device=next(lorita_module.parameters()).device))

            # Process each component
            for i, comp in enumerate(components):
                lorita_linear = getattr(lorita_module, comp)

                weight = torch.eye(in_features, device=next(lorita_module.parameters()).device)

                # Multiply weights in sequence (all but the last layer)
                for j in range(lorita_linear.N - 1):
                    layer = lorita_linear.net1[j]
                    weight = layer.weight.data @ weight

                # Apply final layer
                final_layer = lorita_linear.net1[lorita_linear.N - 1]
                weight = final_layer.weight.data @ weight

                # Set bias if available
                if has_bias and hasattr(final_layer, 'bias') and final_layer.bias is not None:
                    replacement.bias.data[i * dim:(i + 1) * dim] = final_layer.bias.data

                # Set weights for this component (Q, K, or V) in the appropriate section of the QKV matrix
                replacement.weight.data[i * dim:(i + 1) * dim] = weight

        # Replace the module using the handler
        self.replace_module(module_name, replacement, lorita_module)

        return self

    def truncate_module(self, module_name: str, rank: int = 4):
        original_module = self._get_module_by_name(module_name)

        if not isinstance(original_module, nn.Linear):
            raise ValueError(f"Module {module_name} must be a Linear layer")

        self.replace_module(module_name, TruncateLoRALinear(original_module.in_features, original_module.out_features, rank=rank, alpha=1.0, original_weight=original_module.weight.data))

        return self

    def truncate_qkv_module(self, module_name: str, rank: int = 4):
        original_module = self._get_module_by_name(module_name)

        if not isinstance(original_module, nn.Linear):
            raise ValueError(f"Module {module_name} must be a Linear layer")
        self.replace_module(module_name, TruncateLoRAQKVModule(original_module, rank=rank))

        return self

    def remove_weight_regularizer(self, module_name: str):
        """
        Remove a previously added weight regularization hook

        Args:
            module_name (str): Dot-separated path to the module
        """
        if module_name in self.weight_hooks:
            self.weight_hooks[module_name].remove()
            del self.weight_hooks[module_name]

        return self
