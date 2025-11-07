import torch
from torch import tensor


class QuaRS:
    def __init__(self, trainable_weights, target_rank, lmbda, steps, rectangular_mode=False,verbose=False):
        self.trainable_weights = trainable_weights
        self.step = steps
        self.iterations = 0
        # Modified to work directly with tensors
        self.min_dim_layers = [min(layer.shape) for layer in self.trainable_weights]
        self.regularizers = []
        # Access device directly from tensor
        self.val = tensor(0.0).to(self.trainable_weights[0].device)
        self.config = {}
        self.lmbda = lmbda
        self.rectangular_mode = rectangular_mode
        self.target_rank = target_rank
        self.verbose = verbose



        if isinstance(self.target_rank, int) and self.target_rank > 1:
            self.target_ranks = [(int(self.target_rank)) for n in self.min_dim_layers]

        elif isinstance(self.target_rank, list) and len(self.target_rank) == len(self.trainable_weights):
            self.target_ranks = self.target_rank

            # Sort layers by size - now using tensor methods directly
        combined = list(zip(self.target_ranks, self.trainable_weights))
        sorted_combined = sorted(combined, key=lambda x: x[1].numel())
        self.target_ranks, self.trainable_weights = zip(*sorted_combined)

        for target_rank in self.target_ranks:
            self.regularizers.append(
                W_x(epsilon=10 ** 8, target_rank=target_rank, lmbda=lmbda, rectangular_mode=self.rectangular_mode,verbose=verbose))

        print(
            f"QuaRS: Successfully organized {len(trainable_weights)} layers with target ranks: {self.target_ranks}")

    def __call__(self):
        if self.iterations % self.step == 0:
            if self.iterations == 0:
                for i, weight in enumerate(self.trainable_weights):
                    self.regularizers[i].update_weightoperator(weight, epsilon_large_flag=True)
            else:
                for i, weight in enumerate(self.trainable_weights):
                    self.regularizers[i].update_weightoperator(weight)

        self.val = tensor(0.0).to(self.trainable_weights[0].device)

        for i, weight in enumerate(self.trainable_weights):
            if not hasattr(self.regularizers[i], 'U'):
                self.regularizers[i].update_weightoperator(weight)
            self.regularizers[i](weight)

        for weight_operator in self.regularizers:
            self.val += weight_operator.val.to(self.val.device).float()

        self.iterations += 1

        if self.verbose:
            print(f'Target Ranks:           {[wx.target_rank for wx in self.regularizers]}')
            print(f'Epsilon Envelope Rank:  {[wx.epsilon_rank_envelope for wx in self.regularizers]}')
            print(f'Smallest Sigma:         {[wx.smallest_computed_sigma for wx in self.regularizers]}')
            print(f'Length of Values:       {[len(wx.S) for wx in self.regularizers]}')
            print(f'Epsilon:                {[wx.epsilon for wx in self.regularizers]}')
            print(f'Val:                    {[wx.val for wx in self.regularizers]}')
            b=20 * "="
            print(f"{b}\n")

    def calculate_tail_ratio(self):
        svd_ratios = {}
        for i, reg in enumerate(self.regularizers):
            if hasattr(reg, 'S') and reg.S is not None and len(reg.S) > 0:
                singular_values = reg.S
                target_rank = self.target_ranks[i]
                sum_target_rank = torch.sqrt((singular_values[:target_rank] ** 2).sum())
                total_sum = torch.norm(self.trainable_weights[i], p='fro')
                ratio = (sum_target_rank / total_sum)
                svd_ratios[f"Layer {i}"] = ratio.item()
            else:
                # print(f"SVD not initialized or empty for layer {i}")
                svd_ratios[f"Layer {i}"] = None  # You can set None or any other value to indicate it's unavailable
        return svd_ratios

    def update(self):
        self()


class W_x:
    def __init__(self, target_rank, lmbda, epsilon=10 ** 8, verbose=False, rectangular_mode=False):
        super().__init__()
        self.epsilon = epsilon
        self.val = tensor(0.0)  # Initialize as float tensor
        self.smallest_computed_sigma = 0
        self.verbose = verbose  # Set to True for debugging
        self.svd_detail = max(10, target_rank + 2)
        self.target_rank = target_rank
        self.sigma_targetrankplus1 = 0
        self.epsilon_rank_envelope = target_rank
        self.lmbda = lmbda
        self.rectangular_mode = rectangular_mode

    def update_weightoperator(self, weight: torch.Tensor, epsilon_large_flag=False):
        print(f"UpdateWeightOperator Called,{epsilon_large_flag}")
        X = weight.clone()

        try:
            if X.device.type == 'mps':
                self.U, self.S, Vh = torch.linalg.svd(X, full_matrices=False)
                self.V = Vh.T
            else:
                d = min(X.size())

                if epsilon_large_flag:
                    # self.U, self.S, self.V = torch.svd_lowrank(X, q=self.target_rank + 1)
                    self.U, self.S, self.V = torch.svd(X)
                    self.epsilon_rank_envelope = self.target_rank
                else:
                    if self.verbose:
                        torch.manual_seed(42)
                    self.U, self.S, self.V = torch.svd_lowrank(X, q=max(self.target_rank + 1,
                                                                        self.epsilon_rank_envelope))
                    #self.U, self.S, self.V = torch.svd(X)

                self.sigma_targetrankplus1 = self.S[self.target_rank]
                # self.epsilon = min(self.sigma_targetrankplus1.item(),self.epsilon)
                self.epsilon = min(max(self.sigma_targetrankplus1.item(), 1e-8), self.epsilon)

                # print(self.epsilon)
                if epsilon_large_flag == False:
                    if self.S[self.epsilon_rank_envelope - 1] < self.epsilon:
                        self.epsilon_rank_envelope = max(torch.count_nonzero(self.S > self.epsilon).item(), 1)
                        if self.verbose:
                            print(self.S)
                    else:
                        is_envelope_undersized = True
                        while is_envelope_undersized:
                            self.epsilon_rank_envelope += self.target_rank
                            if self.epsilon_rank_envelope >= d:
                                self.epsilon_rank_envelope = d
                                is_envelope_undersized = False
                            self.U, self.S, self.V = torch.svd_lowrank(X, q=max(self.target_rank + 1,
                                                                                self.epsilon_rank_envelope))
                            if self.S[self.epsilon_rank_envelope - 1] < self.epsilon:
                                self.epsilon_rank_envelope = torch.count_nonzero(self.S > self.epsilon).item()
                                is_envelope_undersized = False
                        if self.verbose:
                            print(self.S)
            self.smallest_computed_sigma = self.S[-1]
            self.epsilon_rank_envelope = torch.count_nonzero(self.S > self.epsilon).item()

            self.U = self.U[:, :self.epsilon_rank_envelope].to(
                X.device)  # Ensure SVD components are on the same device as X
            self.S = self.S[:self.epsilon_rank_envelope].to(X.device)
            self.V = self.V[:, :self.epsilon_rank_envelope].to(X.device)
            # print("SVD components initialized successfully")


        except Exception as e:
            print(f"Error during SVD computation: {e}")

            print(f"X shape: {X.shape}")
            print(f"self.target_rank: {self.target_rank}")
            print(f"self.epsilon_rank_envelope: {self.epsilon_rank_envelope}")

    def __call__(self, weight: torch.Tensor):
        # Ensure U, S, and V are properly initialized before calling them
        assert hasattr(self, 'U') and hasattr(self, 'S') and hasattr(self,'V'), "SVD components not initialized. Call `update_weightoperator` first."

        self.U = self.U.to(weight.device)
        self.S = self.S.to(weight.device)
        self.V = self.V.to(weight.device)
        Z = weight

        val = Z.pow(2).sum()
        ZtU = Z.T @ self.U

        ZV = Z @ self.V
        UtZV = self.U.T @ ZV

        if self.rectangular_mode == 2:
            d1 = Z.size(0)
            d2 = Z.size(1)
            if d1 == d2:
                self.rectangular_mode = False

        D = self.epsilon / torch.maximum(self.S, torch.tensor(self.epsilon, device=self.S.device))
        ds = 1 - D
        ZVD = ZV * ds
        ZtUD = ZtU * ds
        Uds = self.U * ds

        if self.rectangular_mode == False:
            ZtUD = ZtU * ds
            T2_2 = Uds.T @ ZVD
            T2 = UtZV * T2_2
            val2 = T2.sum()
            T3 = ZtU * ZtUD
            val3 = -T3.sum()
            T4 = ZV * ZVD
            val4 = -T4.sum()

        elif (self.rectangular_mode == 1) or (self.rectangular_mode == 2):
            print("IM RUNNING!!")
            Dsq = (self.epsilon / torch.maximum(self.S, torch.tensor(self.epsilon, device=self.S.device))) ** 2
            dssq = 1 - Dsq
            if self.rectangular_mode == 1:
                T2a = UtZV * (UtZV * dssq)

                T2b = UtZV * (Dsq * UtZV)

                T2c = (UtZV * D) * (D * UtZV)

                ZtUDsq = ZtU * dssq
                T3 = ZtU * ZtUDsq

                ZVdssq = ZV * dssq
                T4 = ZV * ZVdssq
            elif self.rectangular_mode == 2:
                if d1 < d2:

                    T2a = UtZV * (dssq * UtZV)

                    ZVD = ZV * D
                    T2b = UtZV * (Uds.T @ ZVD)

                    ZtUDsq = ZtU * dssq
                    T3 = ZtU * ZtUDsq

                    T4 = ZV * ZVD
                elif d1 > d2:
                    T2a = UtZV * (UtZV * dssq)

                    UD = self.U * D
                    T2b = UtZV * (UD.T @ ZVD)

                    ZtUD = ZtU * ds
                    T3 = ZtU * ZtUD

                    ZVdssq = ZV * dssq
                    T4 = ZV * ZVdssq
                else:
                    raise ValueError("Issue with size in rectangular_mode == 2.")

            val2a = T2a.sum()
            val2b = -T2b.sum()
            val2 = val2a + val2b
            if self.rectangular_mode == 1:
                val2c = T2c.sum()
                val2 = val2 + val2c
            val3 = -T3.sum()
            val4 = -T4.sum()

        else:
            raise ValueError("Provide either 'False' or the integers 1 or two as value for rectangular_mode.")
        val = val + val2
        val = val + val3
        val = val + val4
        self.val = self.lmbda * val.to(Z.device).float()  # Ensure self.val is a float tensor on the correct device

        # val = val + val4

        # self.val = self.lmbda * val.to(Z.device).float()  # Ensure self.val is a float tensor on the correct device
