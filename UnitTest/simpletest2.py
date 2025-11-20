import numpy as np
import torch
from matplotlib import pyplot as plt

from Functions.AdamQ3R import AdamQ3R
from Functions.EffectiveRank import effective_rank, tail_ratio
from Functions.Q3R import Q3R as QuaRS

w = torch.rand((100, 100), requires_grad=True)

optim = AdamQ3R([w], trainable_weights=[w], target_rank=10, lmbda=0.01)

W_np = w.clone().detach().cpu().numpy()
W_0 = np.linalg.svd(W_np, compute_uv=False)
plt.plot(np.arange(len(W_0)), W_0, label=f"Sigma start")

for i in range(0, 2000):
    optim.zero_grad()
    w.grad = torch.zeros_like(w)
    optim.step()

    print()
    print(effective_rank(w))
    print(tail_ratio(w),10)

W_np = w.clone().detach().cpu().numpy()
W_0 = np.linalg.svd(W_np, compute_uv=False)

plt.plot(np.arange(len(W_0)), W_0, label=f"Sigma end")
plt.legend()
plt.grid(True)
plt.show()
