import torch
import numpy as np
from torch import tensor
from typing import Union, List, Tuple, Dict
import warnings

import torch
from torch import tensor, nn, Tensor
from torch.types import Device
import torch.distributed as dist

import torch
from contextlib import contextmanager

@contextmanager
def preserve_grad_state():
    was_grad_enabled = torch.is_grad_enabled()
    try:
        torch.set_grad_enabled(False)
        yield
    finally:
        torch.set_grad_enabled(was_grad_enabled)

class QuaRS:
    def __init__(self, trainable_modules: Dict[nn.Linear, List[Tuple]], target_rank: int, lmbda: float, steps=5,
                 rectangular_mode=False, scaling=10 ** -3, verbose=False, epsilon_schedule="DEFAULT", N: int = 100000):
        # Distributed setup
        self.distributed = dist.is_initialized()
        if self.distributed:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        self.trainable_modules: Dict[nn.Linear, List[Tuple]] = trainable_modules

        trainable_weights = self.extract_trainable_weights(trainable_modules)

        self.weight_ptrs = [weight.data_ptr() for weight in trainable_weights]

        self.step: int = steps
        self.iterations: int = 0

        self.dim_ratios = [max(layer.shape) / min(layer.shape) for layer in trainable_weights]
        self.regularizers: List[Wx] = []

        self.device: Device = trainable_weights[0].device

        self.val: Tensor = tensor(0.0).to(self.device)

        self.lmbda: float = lmbda
        self.rectangular_mode: bool = rectangular_mode
        self.target_rank = target_rank
        self.scaling = scaling
        self.verbose = verbose
        self.N = N
        self.epsilon_schedule = epsilon_schedule

        # Determine target ranks
        if isinstance(self.target_rank, float) and 0 <= self.target_rank < 1:
            self.target_ranks = [(int(self.target_rank * max(layer.shape))) for layer in trainable_weights]
        elif (isinstance(self.target_rank, int) and self.target_rank > 1) or (
                isinstance(self.target_rank, float) and self.target_rank.is_integer() and self.target_rank > 1):
            self.target_ranks = [int(self.target_rank * ratio) for ratio in self.dim_ratios]
        elif isinstance(self.target_rank, list) and len(self.target_rank) == len(trainable_weights):
            self.target_ranks = self.target_rank

        # Initialize regularizers
        for target_rank in self.target_ranks:
            self.regularizers.append(
                Wx(epsilon=10 ** 8, target_rank=target_rank, lmbda=lmbda, rectangular_mode=self.rectangular_mode,
                   verbose=verbose, device=self.device, N=N, epsilon_schedule=epsilon_schedule
                   ))

        # Pre-allocate gradient buffers and optimization structures
        self._cached_rank_indices = None
        self._grad_buffers_initialized = False
        self.grad_buffers = {}
        self.grad = {}
        self._grad_shapes = []
        self._grad_numel = []
        self._flattened_grad_buffer = None
        
        # Initialize gradient buffers
        for module_reference in self.trainable_modules.keys():
            shape = module_reference.weight.shape
            self.grad_buffers[module_reference] = torch.zeros(
                shape,
                device=self.device,
                dtype=module_reference.weight.dtype
            )
            self._grad_shapes.append(shape)
            self._grad_numel.append(module_reference.weight.numel())
        
        # Single flattened buffer for efficient all-reduce in distributed mode
        if self.distributed and self._grad_numel:
            total_numel = sum(self._grad_numel)
            self._flattened_grad_buffer = torch.zeros(
                total_numel,
                device=self.device,
                dtype=torch.float32
            )
        
        self._grad_buffers_initialized = True
        
        # Validate device placement once
        if trainable_weights and trainable_weights[0].device != self.device:
            raise ValueError(
                "trainable_weights and regularizer are on different devices! "
                "Please make sure to pass initialization the trainable_weights list "
                "after the model has been to the appropriate device!")

        print(f"QuaRS: Successfully organized {len(trainable_weights)} layers with target ranks: {self.target_ranks}")
    
    def _get_my_regularizer_indices(self):
        """Distribute regularizers across ranks"""
        total_regs = len(self.regularizers)
        regs_per_rank = (total_regs + self.world_size - 1) // self.world_size
        start_idx = self.rank * regs_per_rank
        end_idx = min(start_idx + regs_per_rank, total_regs)
        return start_idx, end_idx
    
    @staticmethod
    def extract_trainable_weights(trainable_modules: Dict[nn.Linear, List[Tuple]]):
        """Args:
            trainable_modules: Dict where keys are modules (e.g., nn.Linear) and values are:
                       - None: use full module.weight
                       - List of (start, end) tuples: use slices of module.weight

        Returns:
            List of weight tensors (or slices thereof) to be used for training or regularization.
        """
        weights = []
        for module, slice_list in trainable_modules.items():
            if not hasattr(module, 'weight'):
                continue
            full_weight = module.weight

            if slice_list is None:
                weights.append(full_weight)
            else:
                for start, end in slice_list:
                    if end is None or end == -1:
                        weights.append(full_weight[start:])
                    else:
                        weights.append(full_weight[start:end])
        return weights

    @staticmethod
    def pad_tensor_with_slice_bounds(m_k: torch.Tensor, full_shape: tuple, dim: int, bounds: tuple) -> torch.Tensor:
        """
        Pads `m_k` into a tensor of shape `full_shape` along the specified dimension using flexible bounds.
        """
        if bounds is None:
            return m_k
        start, end = bounds
        full_dim = full_shape[dim]

        # Normalize slice bounds
        start = 0 if start is None else start
        end = full_dim if end in (None, -1) else end

        # Validate shape compatibility
        expected_size = end - start
        if m_k.shape[dim] != expected_size:
            raise ValueError(
                f"Shape mismatch along dimension {dim}: m_k has size {m_k.shape[dim]}, expected {expected_size}")

        # Create empty padded tensor
        padded = torch.zeros(full_shape, dtype=m_k.dtype, device=m_k.device)

        # Build index slices
        index = [slice(None)] * len(full_shape)
        index[dim] = slice(start, end)

        # Assign m_k into padded tensor
        padded[tuple(index)] = m_k
        return padded

    def __call__(self, grad=False):
        """
        Fully optimized with minimal function calls - everything inlined.
        """
        
        trainable_weights = self.extract_trainable_weights(self.trainable_modules)

        # Get rank indices (cached)
        if self.distributed:
            if self._cached_rank_indices is None:
                self._cached_rank_indices = self._get_my_regularizer_indices()
            start_idx, end_idx = self._cached_rank_indices

        # === PHASE 1: Update weight operators (conditionally) ===
        if self.iterations % self.step == 0:
            with torch.no_grad():
                if self.distributed:
                    for i in range(start_idx, end_idx):
                        self.regularizers[i].update_weightoperator(trainable_weights[i])
                else:
                    for i, weight in enumerate(trainable_weights):
                        self.regularizers[i].update_weightoperator(weight)

        # === PHASE 2: Compute regularizer values ===
        with torch.no_grad():
            if self.distributed:
                for i in range(start_idx, end_idx):
                    self.regularizers[i](trainable_weights[i])
            else:
                for i, weight in enumerate(trainable_weights):
                    self.regularizers[i](weight)

        # === PHASE 3: Aggregate values or compute gradients ===
        if not grad:
            # ============ NON-GRADIENT PATH ============
            with torch.no_grad():
                if self.distributed:
                    if end_idx > start_idx:
                        local_vals = torch.stack([
                            self.regularizers[i].val 
                            for i in range(start_idx, end_idx)
                        ])
                        self.val = local_vals.sum().to(self.device, dtype=torch.float32)
                    else:
                        self.val = torch.zeros(1, device=self.device, dtype=torch.float32).squeeze()
                    
                    dist.all_reduce(self.val, op=dist.ReduceOp.SUM)
                else:
                    if len(self.regularizers) > 0:
                        all_vals = torch.stack([reg.val for reg in self.regularizers])
                        self.val = all_vals.to(self.device, dtype=torch.float32).sum()
                    else:
                        self.val = torch.zeros(1, device=self.device, dtype=torch.float32).squeeze()
        
        else:
            # ============ GRADIENT PATH ============
            
            # Zero buffers in-place
            for buffer in self.grad_buffers.values():
                buffer.zero_()
            
            # Compute gradients into buffers
            index = 0
            for module_reference, item in self.trainable_modules.items():
                grad_buffer = self.grad_buffers[module_reference]
                target_dims = item if item is not None else [None]
                
                for weight, truncated_dim in zip(
                    self.extract_trainable_weights({module_reference: item}), 
                    target_dims
                ):
                    # Only compute if assigned to this rank (in distributed mode)
                    if not self.distributed or (start_idx <= index < end_idx):
                        truncated_grad = self.regularizers[index].grad
                        grad_buffer.add_(
                            self.pad_tensor_with_slice_bounds(
                                truncated_grad, 
                                module_reference.weight.shape, 
                                0,
                                truncated_dim
                            )
                        )
                    index += 1
                
                self.grad[module_reference.weight] = grad_buffer
            
            # === Communication and value aggregation ===
            if self.distributed:
                # Single all-reduce for all gradients (flatten -> reduce -> unflatten)
                offset = 0
                for buffer, numel in zip(self.grad_buffers.values(), self._grad_numel):
                    self._flattened_grad_buffer[offset:offset + numel] = buffer.flatten()
                    offset += numel
                
                # Single all-reduce operation
                dist.all_reduce(self._flattened_grad_buffer, op=dist.ReduceOp.SUM)
                
                # Unflatten back to individual buffers
                offset = 0
                for buffer, shape, numel in zip(
                    self.grad_buffers.values(), 
                    self._grad_shapes, 
                    self._grad_numel
                ):
                    buffer.copy_(
                        self._flattened_grad_buffer[offset:offset + numel].reshape(shape)
                    )
                    offset += numel
                
                # Aggregate regularizer values
                with torch.no_grad():
                    if end_idx > start_idx:
                        local_vals = torch.stack([
                            self.regularizers[i].val 
                            for i in range(start_idx, end_idx)
                        ])
                        self.val = local_vals.sum().to(self.device, dtype=torch.float32)
                    else:
                        self.val = torch.zeros(1, device=self.device, dtype=torch.float32).squeeze()
                    
                    dist.all_reduce(self.val, op=dist.ReduceOp.SUM)
            
            else:
                # Non-distributed: just sum values
                with torch.no_grad():
                    all_vals = torch.stack([reg.val for reg in self.regularizers])
                    self.val = all_vals.to(self.device, dtype=torch.float32).sum()

        # Increment iteration counter
        self.iterations += 1

        # Verbose diagnostics
        if self.verbose:
            print(f'Target Ranks:           {[wx.target_rank for wx in self.regularizers]}')
            print(f'Epsilon Envelope Rank:  {[wx.r_env for wx in self.regularizers]}')
            print(f'Smallest Sigma:         {[wx.smallest_computed_sigma for wx in self.regularizers]}')
            print(f'Length of Values:       {[len(wx.S) for wx in self.regularizers]}')
            print(f'Epsilon:                {[wx.epsilon for wx in self.regularizers]}')
            print(f'Val:                    {[wx.val for wx in self.regularizers]}')
            print(f"{'=' * 20}\n")

    def calculate_tail_ratio(self):
        svd_ratios = {}
        trainable_weights = self.extract_trainable_weights(self.trainable_modules)
        for i, reg in enumerate(self.regularizers):
            if hasattr(reg, 'S') and reg.S is not None and len(reg.S) > 0:
                singular_values = reg.S
                target_rank = self.target_ranks[i]
                sum_target_rank = torch.sqrt((singular_values[:target_rank] ** 2).sum())
                total_sum = torch.norm(trainable_weights[i], p='fro')
                ratio = (sum_target_rank / total_sum)
                svd_ratios[f"Layer {i}"] = ratio.item()
            else:
                svd_ratios[f"Layer {i}"] = None
        return svd_ratios

    def update(self):
        try:
            self()
        except Exception as e:
            print(e)
            trainable_weights = self.extract_trainable_weights(self.trainable_modules)
            for layers in trainable_weights:
                print(layers)
            for index, wx in enumerate(self.regularizers):
                print(f"Layer [{index}] Sigma", wx.S)
            for wx in self.regularizers:
                print(wx.rectangular_mode)
            raise e
            
    def state_dict(self):
        """
        Returns the state of the QuaRS object as a dictionary.
        Useful for saving and loading the state of the regularization operator.
        """
        
        if self.trainable_modules:
            warnings.warn(
                "Trainable modules are not saved, as they are part of the model state_dict. "
                "Please ensure to safely reinitialize regularization terms in the same order."
            )
        return {
            'target_ranks': self.target_ranks,
            'lmbda': self.lmbda,
            'step': self.step,
            'rectangular_mode': self.rectangular_mode,
            'scaling': self.scaling,
            'verbose': self.verbose,
            'device': self.device,
            'N': self.N,
            'epsilon_schedule': self.epsilon_schedule,
            'regularizers': [wx.state_dict() for wx in self.regularizers],
            'val': self.val.cpu(),
            'grad': {key: grad.cpu() for key, grad in self.grad.items()},
            'iterations': self.iterations
        }
        
    def load_state_dict(self, state_dict):
        """
        Loads the state of the QuaRS object from a dictionary.
        Useful for restoring the state of the regularization operator.
        """
        self.target_ranks = state_dict['target_ranks']
        self.lmbda = state_dict['lmbda']
        self.step = state_dict['step']
        self.rectangular_mode = state_dict['rectangular_mode']
        self.scaling = state_dict['scaling']
        self.verbose = state_dict['verbose']
        self.N = state_dict['N']
        self.epsilon_schedule = state_dict['epsilon_schedule']

        # Load regularizers
        for wx, wx_state in zip(self.regularizers, state_dict['regularizers']):
            wx.load_state_dict(wx_state)
          
        # Load other attributes
        self.val = state_dict['val'].to(self.device)
        self.grad = {key: grad.to(self.device) for key, grad in state_dict['grad'].items()}
        self.iterations = state_dict['iterations']


class Wx:
    def __init__(self, target_rank, lmbda, device, epsilon=10 ** 8, verbose=False, rectangular_mode=False,
                 N: int = 46875, epsilon_schedule: str = "DEFAULT"):
        self.target_rank = target_rank
        self.lmbda = lmbda

        self.val = tensor(0.0, device=device)  # Initialize as float tensor
        self.smallest_computed_sigma = 0
        self.verbose = verbose  # Set to True for debugging
        self.svd_detail = max(10, target_rank + 2)
        self.target_rank = target_rank
        self.lmbda = lmbda
        self.device = device
        self.epsilon = epsilon
        self.N = N
        self.epsilon_schedule = epsilon_schedule
        self.q = self.target_rank + 1
        self.c = None
        self.first_iter = True

        # stores u,s,v, epsilon

    def __call__(self, weight: torch.Tensor):
        if not hasattr(self, "U"):
            raise ValueError("Update weight operator has not been called yet!")

        self.val = self.lmbda * self.apply_R_x_innerproduct(weight)
        self.grad = self.lmbda * self.compute_quars_gradient(weight)

        return self.val

    def update_weightoperator(self, weight_matrix):
        """
        Updates the reweighting operator R_{W,ε} for a given weight matrix.

        Args:
            weight_matrix (torch.Tensor): Input weight matrix W ∈ ℝ^(d1×d2)
            target_rank (int): Target rank r_target
            epsilon (float): Smoothing parameter ε

        Updates:
            tuple: (r_env, epsilon, U, sigma, V)
                - self.r_env (int): Envelope rank
                - self.epsilon (float): Updated smoothing parameter
                - self.U (torch.Tensor): Left singular vectors
                - self.S (torch.Tensor): Singular values
                - self.V (torch.Tensor): Right singular vectors
        """
        # Compute partial SVD
        # U, sigma, V = torch.svd(weight_matrix)
        with preserve_grad_state():
            if weight_matrix.type() != torch.float32:
                type = weight_matrix.type()
                weight_matrix = weight_matrix.float()
                U, sigma, Vh = torch.linalg.svd(weight_matrix, full_matrices=False)
                V = Vh.T
                weight_matrix = weight_matrix.type(type)
            else:
                U, sigma, Vh = torch.linalg.svd(weight_matrix, full_matrices=False)
                V = Vh.T
            # print(U.shape, sigma.shape, V.shape)

            # Update smoothing parameter epsilon
            if self.epsilon_schedule == 'linear':

                # print(f"Linear epsilon scheduler")

                if self.first_iter:
                    self.first_iter = False
                    self.epsilon = sigma[self.target_rank].item()

                    epsilon_0 = self.epsilon

                    epsilon_n = 10e-6

                    self.c = np.exp((1 / self.N) * (np.log(epsilon_n) - np.log(epsilon_0)))
                else:
                    self.epsilon = max(self.c * self.epsilon, 10e-6)
            elif self.epsilon_schedule == 'constant':

                self.epsilon = self.N

            elif self.epsilon_schedule == "DEFAULT":
                if self.target_rank < len(sigma):
                    self.epsilon = min(self.epsilon, sigma[self.target_rank].item())

            elif self.epsilon_schedule == "exploration":
                self.epsilon = sigma[self.target_rank].item()

            # print(epsilon)
            # Compute envelope rank r(ε,W)
            # r(ε,W) is the number of singular values greater than ε
            r_env = torch.sum(sigma > self.epsilon).item()

            U = U[:, :r_env]
            V = V[:, :r_env]
            sigma = sigma[:r_env]
            self.U = U
            self.V = V
            self.S = sigma
            if len(sigma) >= 1:
                self.smallest_computed_sigma = self.S[-1]
            else:
                self.smallest_computed_sigma = 0
            self.r_env = r_env
            self.epsilon = self.epsilon
            #print(f"finished updating... for {weight_matrix.shape}")

    def apply_R_x_innerproduct(self, X):
        """
        Applies the reweighting operator R_{W,ε}
        """
        device = X.device
        U = self.U.to(device)
        sigma = self.S.to(device)
        V = self.V.to(device)
        # print(V.shape)
        # Compute Σ_ε^(-1) - I
        # Convert scalar 1 to tensor for torch.maximum
        ones = torch.ones_like(sigma)

        # Compute Σ_ε^(-1) - I
        Sigma_epsilon_inv_diag = 1 / torch.maximum(sigma / self.epsilon, ones)
        Sigma_epsilon_inv = torch.diag(Sigma_epsilon_inv_diag)
        # (Sigma_epsilon^{-1} - I)
        sigma_eps_inv_minus_I = Sigma_epsilon_inv - torch.eye(len(Sigma_epsilon_inv_diag), device=device)

        # Rest of the function remains the same...
        term1 = torch.trace(X.t() @ X)

        UtW = U.t() @ X
        VtWtU = V.t() @ X.t() @ U
        temp = VtWtU @ sigma_eps_inv_minus_I
        term2 = torch.trace(temp @ UtW @ V @ sigma_eps_inv_minus_I)

        term3 = torch.trace(V.t() @ X.t() @ X @ V @ sigma_eps_inv_minus_I)
        term4 = torch.trace(U.t() @ X @ X.t() @ U @ sigma_eps_inv_minus_I)

        return term1 + term2 + term3 + term4

    def compute_quars_gradient(self, X):
        """
        Memory-efficient computation of QUARS regularization gradient
        Handles both single component dict and list of component dicts
        """

        """U = self.U
        sigma = self.S
        V = self.V
        epsilon = self.epsilon

        original_shape = X.shape

        weight_matrix = X
        device = X.device

        ones = torch.ones_like(sigma)
        Sigma_epsilon_inv_diag = 1 / torch.maximum(sigma / epsilon, ones)
        Sigma_epsilon_inv = torch.diag(Sigma_epsilon_inv_diag)

        UU_T = U @ U.T
        VV_T = V @ V.T

        I_d1 = torch.eye(weight_matrix.shape[0], device=device)
        I_d2 = torch.eye(weight_matrix.shape[1], device=device)

        term1 = U @ Sigma_epsilon_inv @ U.T @ weight_matrix @ V @ Sigma_epsilon_inv @ V.T
        term2 = U @ Sigma_epsilon_inv @ U.T @ weight_matrix @ (I_d2 - VV_T)
        term3 = (I_d1 - UU_T) @ weight_matrix @ V @ Sigma_epsilon_inv @ V.T
        term4 = (I_d1 - UU_T) @ weight_matrix @ (I_d2 - VV_T)

        grad = term1 + term2 + term3 + term4"""
        U, V = self.U, self.V
        sigma_eps = self.S
        epsilon = self.epsilon

        # Create identity matrices
        """I_U = torch.eye(U.size(0), device=self.device)
        I_V = torch.eye(V.size(0), device=self.device)

        # Compute sigma_eps_inv
        sigma_eps_inv = 1.0 / (sigma_eps + epsilon)

        # Compute gradient terms
        Z = X
        term1 = U @ (sigma_eps_inv.unsqueeze(-1) * (U.T @ Z @ V)) @ (sigma_eps_inv.unsqueeze(-1) * V.T)
        term2 = U @ (sigma_eps_inv.unsqueeze(-1) * (U.T @ Z @ (I_V - V @ V.T)))
        term3 = (I_U - U @ U.T) @ Z @ V @ (sigma_eps_inv.unsqueeze(-1) * V.T)
        term4 = (I_U - U @ U.T) @ Z @ (I_V - V @ V.T)"""

        ones = torch.ones_like(sigma_eps)

        Sigma_epsilon_inv_diag = 1 / torch.maximum(sigma_eps / epsilon, ones)

        Sigma_epsilon_inv = torch.diag(Sigma_epsilon_inv_diag)

        UU_T = U @ U.T

        VV_T = V @ V.T

        I_d1 = torch.eye(X.shape[0], device=self.device)

        I_d2 = torch.eye(X.shape[1], device=self.device)

        term1 = U @ Sigma_epsilon_inv @ U.T @ X @ V @ Sigma_epsilon_inv @ V.T

        term2 = U @ Sigma_epsilon_inv @ U.T @ X @ (I_d2 - VV_T)

        term3 = (I_d1 - UU_T) @ X @ V @ Sigma_epsilon_inv @ V.T

        term4 = (I_d1 - UU_T) @ X @ (I_d2 - VV_T)

        # Sum all terms
        grad = term1 + term2 + term3 + term4

        # grad = grad.view(original_shape)

        return 2 * grad
    def state_dict(self):
        """
        Returns the state of the Wx object as a dictionary.
        Useful for saving and loading the state of the regularization operator.
        """
        state = {}
        if not hasattr(self, 'U'):
            return None
        # List of attributes to save with their default values if missing
        attributes = {
            'target_rank': None,
            'lmbda': None,
            'val': None,
            'smallest_computed_sigma': None,
            'verbose': False,
            'svd_detail': None,
            'device': None,
            'epsilon': None,
            'N': None,
            'epsilon_schedule': None,
            'r_env': None,
            'first_iter': True,
            'q': None,
            'c': None
        }
        
        for attr_name, default_value in attributes.items():
            state[attr_name] = getattr(self, attr_name, default_value)
        
        # Handle tensor attributes that need .cpu() - check if they exist and are not None
        for tensor_attr in ['U', 'S', 'V']:
            if hasattr(self, tensor_attr) and getattr(self, tensor_attr) is not None:
                state[tensor_attr] = getattr(self, tensor_attr).cpu()
            else:
                state[tensor_attr] = None
        
        return state

    def load_state_dict(self, state_dict):
        """
        Loads the state of the Wx object from a dictionary.
        Useful for restoring the state of the regularization operator.
        """
        if state_dict is None:
            return # Nothing to load
        self.target_rank = state_dict['target_rank']
        self.lmbda = state_dict['lmbda']
        self.val = state_dict['val'].to(self.device)
        self.smallest_computed_sigma = state_dict['smallest_computed_sigma']
        self.verbose = state_dict['verbose']
        self.svd_detail = state_dict['svd_detail']
        self.epsilon = state_dict['epsilon']
        self.N = state_dict['N']
        self.epsilon_schedule = state_dict['epsilon_schedule']
        self.first_iter = state_dict['first_iter']
        self.q = state_dict['q']
        self.c = state_dict['c']

        # Load U, S, V
        self.U = state_dict['U'].to(self.device)
        self.S = state_dict['S'].to(self.device)
        self.V = state_dict['V'].to(self.device)
        
        # Load r_env
        self.r_env = state_dict['r_env']
