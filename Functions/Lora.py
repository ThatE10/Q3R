from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import math


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("Rank must be a positive integer.")
        if 0 <= rank < 1:
            rank = int(math.floor((rank*in_features*out_features)/(in_features + out_features)))
        if isinstance(rank, float) and rank.is_integer():
            rank = int(rank)
        if not isinstance(rank, int):
            raise ValueError("Rank must be an integer.")
        if rank > min(in_features, out_features):
            raise ValueError("Rank must be less than or equal to min(in_features, out_features).")
        if alpha <= 0:
            raise ValueError("Alpha must be a positive float.")
        if not isinstance(alpha, float):
            raise ValueError("Alpha must be a float.")
        if alpha > 1:
            raise ValueError("Alpha must be less than or equal to 1.")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = int(rank)
        self.alpha = alpha

        # Low-rank decomposition A (in_features × rank) and B (rank × out_features)
        self.A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, out_features) * 0.01)

        # Scaling factor
        self.scaling = alpha / rank

    def forward(self, x):
        return x @ self.A @ self.B * self.scaling

    def __repr__(self):
        # Return a string representation of the instance with the parameters (in_features, out_features, rank)
        return f"LoRALinear(in_features={self.in_features}, out_features={self.out_features}, rank={self.rank})"


class TruncateLoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0, original_weight=None):
        super().__init__()
        if original_weight is None:
            raise ValueError("original_weight must be provided for SVD-based initialization.")
        if rank <= 0:
            raise ValueError("Rank must be a positive integer.")

        if 0 <= rank < 1:
            rank = int(math.floor((rank * in_features * out_features) / (in_features + out_features)))

        if not isinstance(rank, int):
            raise ValueError("Rank must be an integer.")
        if rank > min(in_features, out_features):
            raise ValueError("Rank must be less than or equal to min(in_features, out_features).")
        if alpha <= 0:
            raise ValueError("Alpha must be a positive float.")
        if not isinstance(alpha, float):
            raise ValueError("Alpha must be a float.")
        if alpha > 1:
            raise ValueError("Alpha must be less than or equal to 1.")

        self.rank = int(rank)
        self.alpha = alpha

        A_init, B_init = self._initialize_lora_weights(original_weight, rank)

        self.A = nn.Parameter(A_init)
        self.B = nn.Parameter(B_init)
        self.in_features = in_features
        self.rank = rank
        self.out_features = out_features

    @staticmethod
    def _initialize_lora_weights(weight: Tensor, rank: int) -> Tuple[Tensor, Tensor]:
        """Performs truncated SVD to initialize LoRA weights."""
        weight = weight.view(weight.shape[0], -1)  # Ensure 2D
        try:
            U, S, Vt = torch.linalg.svd(weight, full_matrices=False)
        except RuntimeError:
            U, S, Vt = map(torch.tensor, np.linalg.svd(weight.detach().cpu().numpy(), full_matrices=False))

        k = min(rank, len(S))
        
        #sanity check if weight is equivalent to the original weight
        # weight - svd(weight) should be close to zero
        #print(torch.allclose(weight, (U * S.unsqueeze(-1)) @ Vt, atol=1e-6))

        print("Warning: Original weight is equivalent to the SVD decomposition. Consider using a different rank or original weight.")
        
        return U[:, :k] * S[:k].sqrt(), (S[:k].sqrt().unsqueeze(-1) * Vt[:k, :])
        #return U * S.sqrt(), (S.sqrt().unsqueeze(-1) * Vt)

    def forward(self, x):
        return (x @ self.B.T) @ self.A.T

    def __repr__(self):
        # Return a string representation of the instance with the parameters (in_features, out_features, rank)
        return f"TruncateLoRA(in_features={self.in_features}, out_features={self.out_features}, rank={self.rank})"


class LoRAQKV(nn.Module):
    def __init__(self, original_qkv, rank=4, alpha=1.0):
        super().__init__()

        # Get original weight dimensions
        total_dim = original_qkv.weight.shape[0]
        dim = total_dim // 3  # Assuming Q, K, V are concatenated

        # LoRA layers for Q, K, V
        self.q_lora = LoRALinear(original_qkv.in_features, dim, rank, alpha)
        self.k_lora = LoRALinear(original_qkv.in_features, dim, rank, alpha)
        self.v_lora = LoRALinear(original_qkv.in_features, dim, rank, alpha)

    def forward(self, x):

        q = self.q_lora(x)
        k = self.k_lora(x)
        v = self.v_lora(x)

        return torch.cat([q, k, v], dim=-1)


class TruncateLoRAQKVModule(nn.Module):
    def __init__(self, original_qkv, rank=4, alpha=1.0):
        super().__init__()

        # Get original weight dimensions
        total_dim = original_qkv.weight.shape[0]
        dim = total_dim // 3  # Assuming Q, K, V are concatenated

        # LoRA layers for Q, K, V with SVD-based weight initialization
        self.q_lora = TruncateLoRALinear(original_qkv.in_features, dim, rank, alpha, original_qkv.weight[:dim])
        self.k_lora = TruncateLoRALinear(original_qkv.in_features, dim, rank, alpha, original_qkv.weight[dim:2 * dim])
        self.v_lora = TruncateLoRALinear(original_qkv.in_features, dim, rank, alpha, original_qkv.weight[2 * dim:])

    def forward(self, x):
        # Original QKV transformation

        # LoRA adaptations for Q, K, V
        q = self.q_lora(x)
        k = self.k_lora(x)
        v = self.v_lora(x)

        # Reconstruct output
        return torch.cat([q, k, v], dim=-1)
