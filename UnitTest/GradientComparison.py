
import torch

from Functions.Q3R import Wx

# Set seed and dimensions
torch.manual_seed(0)
n, d, r = 10, 10, 4

X = torch.randn(n, d, requires_grad=True)
U, sigma, V = torch.svd(X)

epsilon = 0.1
ones = torch.ones_like(sigma)

sigma_eps_inv_diag = 1.0 / torch.maximum(sigma / epsilon, ones)
Sigma_epsilon_inv = torch.diag(sigma_eps_inv_diag)
delta = Sigma_epsilon_inv - torch.eye(10)

# Compute function
Xt = X.T
Ut = U.T
Vt = V.T

term1 = torch.trace(Xt @ X)
A = Vt @ Xt @ U
term2 = torch.trace((A @ delta) @ (A.T @ delta))
term3 = torch.trace(Vt @ Xt @ X @ V @ delta)
term4 = torch.trace(Ut @ X @ Xt @ U @ delta)
f = term1 + term2 + term3 + term4

# Autograd gradient
f.backward()
autograd_grad = X.grad.clone()

# Manually computed gradient
delta2 = delta @ delta

Wx(epsilon=10 ** 8, target_rank=10, lmbda=0.01,
                   verbose=True, device=X.device, N=46875, epsilon_schedule="DEFAULT")

# Compare
print("Autograd Gradient:\n", autograd_grad)
print("Manual Gradient:\n", manual_grad)
print("Close match:", torch.allclose(autograd_grad, manual_grad, atol=1e-5))
