from typing import List, Union

import torch
from torch import nn, tensor


class Hoyer:
    def __init__(self, trainable_weights: Union[torch.Tensor, nn.Linear]):
        self.device: Device = trainable_weights[0].device
        self.trainable_weights: List[torch.Tensor] = [layer if type(layer) == torch.Tensor else layer.weight for layer in trainable_weights]
        self.val = tensor(0.0).to(self.device)


    def __call__(self):
        self.val = tensor(0.0).to(self.device)
        for layer in self.trainable_weights:
            self.val+=Reg_Loss(layer,device=self.device)

    def update(self):
        self()


def Reg_Loss(weight: torch.Tensor, device, reg_type='Hoyer')-> float:
    reg = torch.tensor(0.0).to(device)
    u,s,v = torch.svd(weight)

    if reg_type == "Hoyer":
        reg += torch.sum(torch.abs(s)) / torch.sqrt(torch.sum(s ** 2)) - 1
    elif reg_type == "Hoyer-Square":
        reg += (torch.sum(torch.abs(s)) ** 2) / torch.sum(s ** 2) - 1
    elif reg_type == "L1":
        reg += torch.sum(torch.abs(s))
    else:
        reg = 0.0.to(device)

    return reg