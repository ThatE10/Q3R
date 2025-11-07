import torch
from torch import nn


def effective_rank(layer_or_tensor):
    # Get the weight matrix safely
    if isinstance(layer_or_tensor, nn.Linear):
        matrix = layer_or_tensor.weight.data
    elif isinstance(layer_or_tensor, torch.Tensor):
        matrix = layer_or_tensor
    elif hasattr(layer_or_tensor, 'weight'):
        matrix = layer_or_tensor.weight.data
    else:
        raise TypeError("Input must be a torch.Tensor, nn.Linear, or object with a `.weight` attribute.")

    # Make sure it's at least 2D
    if matrix.ndim != 2:
        raise ValueError("Input matrix must be 2D.")

    # Use SVD to get singular values
    try:
        _, S, _ = torch.linalg.svd(matrix.to(dtype=torch.float64, device="cpu"), full_matrices=False)
    except RuntimeError as e:
        raise RuntimeError(f"SVD failed: {e}")

    if S.numel() == 0 or S.sum() == 0:
        return 0.0

    # Normalize singular values to get a probability distribution
    p = S / S.sum()

    # Compute entropy only on non-zero entries
    p_nonzero = p[p > 0]
    entropy = -torch.sum(p_nonzero * torch.log(p_nonzero))

    return entropy.item()


def tail_ratio(layer_or_tensor, r=1,S=None):
    """
    Calculate the tail ratio: S[:r].sum() / (Frobenius norm)^2

    Args:
        layer_or_tensor: A PyTorch layer, tensor, or object with a weight attribute
        r: Number of top singular values to use (default: 1)

    Returns:
        The tail ratio as a float
    """
    # Get the weight matrix safely
    if isinstance(layer_or_tensor, nn.Linear):
        matrix = layer_or_tensor.weight.data
    elif isinstance(layer_or_tensor, torch.Tensor):
        matrix = layer_or_tensor
    elif hasattr(layer_or_tensor, 'weight'):
        matrix = layer_or_tensor.weight.data
    else:
        raise TypeError("Input must be a torch.Tensor, nn.Linear, or object with a `.weight` attribute.")

    # Make sure it's at least 2D
    if matrix.ndim != 2:
        raise ValueError("Input matrix must be 2D.")

    # Use SVD to get singular values
    if S is None:
        try:
            _, S, _ = torch.linalg.svd(matrix.to(dtype=torch.float64, device="cpu"), full_matrices=False)
        except RuntimeError as e:
            raise RuntimeError(f"SVD failed: {e}")

        if S.numel() == 0 or S.sum() == 0:
            return 0.0
    

    # Calculate Frobenius norm squared (sum of squared singular values)
    frob_norm_squared = torch.sum(S ** 2)

    # Ensure r is valid
    r = min(r, len(S))

    # Calculate tail ratio: sum of top r singular values divided by Frobenius norm squared
    ratio = torch.sum(S[:r] ** 2) / frob_norm_squared

    return ratio.item()