"""
Working AdamQ3R optimizer with QUARS regularization
"""

from typing import Union, Dict, List, Tuple

import torch
from torch import nn
from torch.optim import Optimizer

from Functions.Q3R import Q3R


class AdamQ3R(Optimizer):
    def __init__(self, params, trainable_modules: Dict[nn.Linear, List[Tuple]], target_rank: Union[float, int],
                 lr=0.001, betas=(0.9, 0.999), eps=1e-8, lmbda=0.1, schedule_fn=None, N=46875,
                 epsilon_schedule="DEFAULT", steps=5, verbose = False):
        """
        Implementation of Adam with QUARS regularization.
        
        This optimizer integrates the Q3R (Quadratic Reweighted Rank Regularizer) directly into the Adam update step.
        It supports regularization of standard linear layers as well as fused layers (e.g., QKV projections) by
        accepting a `trainable_modules` dictionary that can specify slices of weight matrices.

        Args:
            params: iterable of parameters to optimize
            trainable_weights: a list of linear layers or torch weight references objects
            target_rank: [0<x<1] for percent weight reduction or an integer value, r < min(M,n), which is scaled by max(m,n)/min(n,m)
            lr: learning rate α (default: 0.001)
            betas: coefficients for moving averages (default: (0.9, 0.999))
            eps: term for numerical stability (default: 1e-8)
            lmbda: Q3R regularization parameter (default: 0.1)
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

        self.q3r = Q3R(trainable_modules=trainable_modules, target_rank=target_rank, lmbda=lmbda,
                         verbose=verbose, N=N,
                         epsilon_schedule=epsilon_schedule, steps=steps)

        self.schedule_fn = schedule_fn
        super().__init__(params, defaults)

    def step(self, closure=None):
        self.q3r(grad=True)
        """
        Performs a single optimization step with Q3R regularization

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

                mt, vt = state['mt'], state['vt']  #retrieve m_(t-1) & v_(t-1)
                beta1, beta2 = group['beta1'], group['beta2']
                lmbda = group['lmbda']
                lr = group['lr']

                state['step'] += 1
                t = state['step']

                gt = grad #+ lr * p.data #\STATE $\f{g}_t \gets \nabla f_t(\f{W}_{t-1}) $

                mt.mul_(beta1).add_(gt, alpha=1 - beta1)
                vt.mul_(beta2).addcmul_(gt, gt, value=1 - beta2) #$\f{m}_t \gets \beta_1 \f{m}_{t-1} + (1 - \beta_1) \f{g}_t$


                mt_hat = mt / (1 - beta1 ** t)
                vt_hat = vt / (1 - beta2 ** t)


                denom = vt_hat.sqrt().add_(group['eps']) # \sqrt{\hat{\mathbf{v}}_t} + \epsilon
                #p.data.addcdiv_(mt_hat, denom, value=-lr)
                p.data.addcdiv_(mt_hat, denom, value=-lr)

                if p in self.q3r.grad.keys():
                    p.data.add_( eta_t * self.q3r.grad[p], alpha=-lr) #q3r includes regularisation lambda
        
        return loss
    def state_dict(self):
        """
        Returns the state of the optimizer as a dictionary
        """
        return {
            'state': self.state,
            'q3r_state': self.q3r.state_dict()
        }
    def load_state_dict(self, state_dict):
        """
        Loads the optimizer state from a dictionary
        """
        self.state = state_dict['state']
        self.q3r.load_state_dict(state_dict['q3r_state'])
