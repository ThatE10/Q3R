import unittest

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import Optimizer

from Functions.AdamQ3R import AdamQ3R as AdamQ3REthan
from Functions.quars_v3 import QuaRS


def update_reweighting_operator(weight_matrix, target_rank, epsilon):
    """
    Updates the reweighting operator R_{W,ε} for a given weight matrix.

    Args:
        weight_matrix (torch.Tensor): Input weight matrix W ∈ ℝ^(d1×d2)
        target_rank (int): Target rank r_target
        epsilon (float): Smoothing parameter ε

    Returns:
        tuple: (r_env, epsilon, U, sigma, V)
            - r_env (int): Envelope rank
            - epsilon (float): Updated smoothing parameter
            - U (torch.Tensor): Left singular vectors
            - sigma (torch.Tensor): Singular values
            - V (torch.Tensor): Right singular vectors
    """
    # Compute partial SVD
    # U, sigma, V = torch.svd(weight_matrix)
    U, sigma, Vh = torch.linalg.svd(weight_matrix, full_matrices=False)
    V = Vh.T
    # print(U.shape, sigma.shape, V.shape)

    # Update smoothing parameter epsilon
    if target_rank < len(sigma):
        epsilon = min(epsilon, sigma[target_rank].item())
    # print(epsilon)
    # Compute envelope rank r(ε,W)
    # r(ε,W) is the number of singular values greater than ε
    r_env = torch.sum(sigma > epsilon).item()

    # Select first r_env components
    U_full = U
    V_full = V
    sigma_full = sigma
    U = U[:, :r_env]
    V = V[:, :r_env]
    sigma = sigma[:r_env]

    return r_env, epsilon, U, sigma, V


def compute_quars_gradient(weight_matrix, r_env, epsilon, device, lmbda):
    """
    Memory-efficient computation of QUARS regularization gradient
    Handles both single component dict and list of component dicts
    """

    """ if isinstance(components, list):
        matching_comp = None
        for comp in components:
            if weight_matrix.shape == comp['shape']:  # Match using stored shape
                matching_comp = comp
                break
        if matching_comp is None:
            # Skip if no matching component found
            return torch.zeros_like(weight_matrix)
        U, V = matching_comp['U'], matching_comp['V']
        sigma_eps = matching_comp['sigma']
        epsilon = matching_comp['eps']
    else:
        U, V = components['U'], components['V']
        sigma_eps = components['sigma']
        epsilon = components['eps']"""
    U, sigma, V = torch.svd(weight_matrix)
    U = U[:, :r_env]
    V = V[:, :r_env]
    sigma_eps = sigma[:r_env]

    """original_shape = weight_matrix.shape

    weight_matrix = weight_matrix.to(device)
    U = U.to(device)
    V = V.to(device)
    sigma_eps = sigma.to(device)

    ones = torch.ones_like(sigma_eps)
    Sigma_epsilon_inv_diag = 1 / torch.maximum(sigma_eps / epsilon, ones)
    Sigma_epsilon_inv = torch.diag(Sigma_epsilon_inv_diag)

    UU_T = U @ U.T
    VV_T = V @ V.T
    I_d1 = torch.eye(weight_matrix.shape[0], device=device)
    I_d2 = torch.eye(weight_matrix.shape[1], device=device)

    term1 = U @ Sigma_epsilon_inv @ U.T @ weight_matrix @ V @ Sigma_epsilon_inv @ V.T
    term2 = U @ Sigma_epsilon_inv @ U.T @ weight_matrix @ (I_d2 - VV_T)
    term3 = (I_d1 - UU_T) @ weight_matrix @ V @ Sigma_epsilon_inv @ V.T
    term4 = (I_d1 - UU_T) @ weight_matrix @ (I_d2 - VV_T)

    grad = term1 + term2 + term3 + term4
    grad = grad.view(original_shape)"""

    ones = torch.ones_like(sigma_eps)

    Sigma_epsilon_inv_diag = 1 / torch.maximum(sigma_eps / epsilon, ones)

    Sigma_epsilon_inv = torch.diag(Sigma_epsilon_inv_diag)

    UU_T = U @ U.T

    VV_T = V @ V.T

    I_d1 = torch.eye(weight_matrix.shape[0], device=device)

    I_d2 = torch.eye(weight_matrix.shape[1], device=device)

    term1 = U @ Sigma_epsilon_inv @ U.T @ weight_matrix @ V @ Sigma_epsilon_inv @ V.T

    term2 = U @ Sigma_epsilon_inv @ U.T @ weight_matrix @ (I_d2 - VV_T)

    term3 = (I_d1 - UU_T) @ weight_matrix @ V @ Sigma_epsilon_inv @ V.T

    term4 = (I_d1 - UU_T) @ weight_matrix @ (I_d2 - VV_T)

    return 2 * lmbda * (term1 + term2 + term3 + term4)


class AdamQ3RIpsita(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, lmbda=0.1, schedule_fn=None):
        """
        Implementation of Adam with QUARS regularization

        Args:
            params: iterable of parameters to optimize
            lr: learning rate α (default: 0.001)
            betas: coefficients for moving averages (default: (0.9, 0.999))
            eps: term for numerical stability (default: 1e-8)
            quars_lambda: QUARS regularization parameter (default: 0.1)
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")

        defaults = dict(
            lr=lr,
            beta1=betas[0],
            beta2=betas[1],
            eps=eps,
            lmbda=lmbda
        )
        self.schedule_fn = schedule_fn
        super().__init__(params, defaults)

    def step(self, closure=None, parmeters=None):
        """
        Performs a single optimization step with QUARS regularization

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss
            quars_components: Dictionary containing QUARS components (U, V, sigma, epsilon)
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['mt'] = torch.zeros_like(p.data)
                    state['vt'] = torch.zeros_like(p.data)

                mt, vt = state['mt'], state['vt']
                beta1, beta2 = group['beta1'], group['beta2']
                lmbda = group['lmbda']
                lr = group['lr']

                state['step'] += 1
                t = state['step']

                gt = grad #+ lr * p.data

                mt.mul_(beta1).add_(gt, alpha=1 - beta1)
                vt.mul_(beta2).addcmul_(gt, gt, value=1 - beta2)

                mt_hat = mt / (1 - beta1 ** t)
                vt_hat = vt / (1 - beta2 ** t)

                eta_t = 3

                denom = vt_hat.sqrt().add_(group['eps'])
                p.data.addcdiv_(mt_hat, denom, value=-eta_t * lr)
                if parmeters is not None and p in parmeters.keys():
                    quars_grad = compute_quars_gradient(
                        p,
                        parmeters[p]['r_env'], parmeters[p]['epsilon'],
                        p.device,
                        lmbda
                    )
                    p.data.add_(eta_t * quars_grad, alpha=-lr)

        return loss


class RegularisationConversionUnitTest(unittest.TestCase):
    class BasicModel(nn.Module):
        def __init__(self, input_size, output_size, layers=1):
            super().__init__()

            if layers == 1:
                self.model = nn.Sequential(*[nn.Linear(input_size, output_size)])
            elif layers == 2:
                self.model = nn.Sequential(*[nn.Linear(input_size, 100), nn.Linear(100, output_size)])
            else:
                if layers <= 2:
                    raise ValueError("Layers cannot be less than 1")

                model_list = [nn.Linear(input_size, 100)] + [nn.Linear(100, 100) for i in range(layers - 2)] + [
                    nn.Linear(100, output_size)]
                self.model = nn.Sequential(*model_list)

        def forward(self, x):
            x = self.model(x)
            return x

    class BasicRectangularModel(nn.Module):
        def __init__(self, input_size, output_size=200, layers=1):
            super().__init__()

            if layers == 1:
                self.model = nn.Sequential(*[nn.Linear(input_size, output_size)])
            elif layers == 2:
                self.model = nn.Sequential(*[nn.Linear(input_size, 200), nn.Linear(200, output_size)])
            else:
                if layers <= 2:
                    raise ValueError("Layers cannot be less than 1")

                model_list = [nn.Linear(input_size, 200)] + [nn.Linear(200, 100) if i % 2 == 0 else nn.Linear(100, 200)
                                                             for i in range(layers - 2)] + [
                                 nn.Linear(200, output_size) if layers % 2 == 0 else nn.Linear(100, output_size)]
                self.model = nn.Sequential(*model_list)

        def forward(self, x):
            x = self.model(x)
            return x

    class BasicModelOld(nn.Module):
        def __init__(self, input_size, output_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, output_size)

        def forward(self, x):
            x = self.fc1(x)
            return x

    def test_square_convergence_single_layer(self):
        LR = 0.1
        lmbda = 0.01
        dim = (100, 100)
        target_rank = 10
        layers = 1
        for j in range(1):
            torch.manual_seed(j)

            model_A, model_B, model_C = self.BasicModel(dim[0], dim[1], layers=1), self.BasicModel(dim[0], dim[1],
                                                                                                   layers=1), self.BasicModel(
                dim[0], dim[1], layers=1)

            for index in range(layers):
                init_matrix = torch.rand(dim)
                model_A.model[index].weight.data = init_matrix.clone()
                model_B.model[index].weight.data = init_matrix.clone()
                model_C.model[index].weight.data = init_matrix.clone()
            print(model_A, model_B, model_C)

            for index, layer in enumerate(model_A.model):
                A_s = torch.svd(layer.weight.data)[1]
                plt.plot(np.arange(len(A_s)), A_s, label=f"Quars Model A Layer:{index}")

            for index, layer in enumerate(model_B.model):
                B_s = torch.svd(layer.weight.data)[1]
                plt.plot(np.arange(len(B_s)), B_s, label=f"Ethan Model B Layer:{index}")

            for index, layer in enumerate(model_B.model):
                C_s = torch.svd(layer.weight.data)[1]
                plt.plot(np.arange(len(B_s)), C_s, label=f"Ipsita Model B Layer:{index}")

            old_q3r = QuaRS(trainable_modules={layer:None for layer in model_A.model}, target_rank=target_rank,
                            lmbda=lmbda, steps=1,
                            rectangular_mode=False, verbose=False)

            optimizer_A = torch.optim.SGD(model_A.parameters(), lr=LR)
            optimizer_B = AdamQ3REthan(model_B.parameters(),
                                       lr=LR,
                                       trainable_modules={layer:None for layer in model_B.model},
                                       target_rank=target_rank,
                                       lmbda=lmbda,
                                       steps=1,verbose=False)



            optimizer_C = AdamQ3RIpsita(model_C.parameters(),
                                        lr=LR,
                                        lmbda=lmbda
                                        )

            optimizer_C_parameters = {layer.weight: {'epsilon': 10 ** 8} for layer in model_C.model}

            for i in range(1000):
                optimizer_A.zero_grad(), optimizer_B.zero_grad(), optimizer_C.zero_grad()
                print("old"), old_q3r.update()

                old_q3r.val.backward()

                optimizer_A.step()

                # ========================================================================================================#
                for p in model_B.parameters():
                    if p.requires_grad and p.data_ptr():# in optimizer_B.q3r.grad.keys():
                        p.grad = torch.zeros_like(p)

                

                optimizer_B.step()


                # ========================================================================================================#
                for p in model_C.parameters():
                    if p.requires_grad:
                        p.grad = torch.zeros_like(p)

                optimizer_C_parameters = {layer.weight: {'r_env': update_reweighting_operator(layer.weight.data, target_rank,
                                                                                       optimizer_C_parameters[layer.weight][
                                                                                           'epsilon'])[0],
                                                  'epsilon':
                                                      update_reweighting_operator(layer.weight.data, target_rank,
                                                                                  optimizer_C_parameters[layer.weight][
                                                                                      'epsilon'])[1]}
                                          for layer in
                                          model_C.model}

                optimizer_C.step(parmeters=optimizer_C_parameters)

            for index, layer in enumerate(model_A.model):
                A_s = torch.svd(layer.weight.data)[1]
                plt.plot(np.arange(len(A_s)), A_s, label=f"Model A Layer:{index}")

            for index, layer in enumerate(model_B.model):
                B_s = torch.svd(layer.weight.data)[1]
                plt.plot(np.arange(len(B_s)), B_s, label=f"Model B Layer:{index}")

            for index, layer in enumerate(model_C.model):
                C_s = torch.svd(layer.weight.data)[1]
                plt.plot(np.arange(len(C_s)), C_s, label=f"Model C Layer:{index}")

            plt.legend()
            plt.grid(True)
            plt.show()
