import torch
from torch import nn


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

model = BasicModel(100,10, 3)
model.layer_thing = model.model[0].weight[:5]
print(id(model.layer_thing))
print(id( model.model[0].weight[:5]))
print(model.model[0].weight[:5].data_ptr())
for p in model.parameters():
    #print(id(p))
    #print(p.shape)
    print(p.data_ptr())
