import numpy as np
import torch
from matplotlib import pyplot as plt

from Functions.EffectiveRank import effective_rank, tail_ratio
from Functions.quars_v3 import QuaRS

w = torch.rand((100, 100))

reg = QuaRS(trainable_weights=[w], target_rank=10, lmbda=1)


def apply_grad(M, lr):
    reg(grad=True)
    print(f'epsilon: {reg.regularizers[0].epsilon}')
    grad = reg.regularizers[0].grad
    M -= lr * grad
    return M


W_0 = torch.svd(w)[1]
plt.plot(np.arange(len(W_0)), W_0, label=f"Sigma start")

for i in range(0, 2000):
    w = apply_grad(w, lr=0.001)
    print()
    print(effective_rank(w))
    print(tail_ratio(w))

W_0 = torch.svd(w)[1]
plt.plot(np.arange(len(W_0)), W_0, label=f"Sigma end")
plt.legend()
plt.grid(True)
plt.show()
