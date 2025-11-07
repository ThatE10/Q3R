import torch
import torch.nn as nn
import torch.nn.functional as F

from Functions.Lora import LoRALinear


class DNN(nn.Module):
    def __init__(self, IMAGE_SIZE, CHANNELS, NUM_CLASSES):
        super().__init__()
        self.input = nn.Linear(IMAGE_SIZE * IMAGE_SIZE * CHANNELS, 784)
        self.fc2 = nn.Linear(784, 784)
        self.fc3 = nn.Linear(784, 300)
        self.output = nn.Linear(300, NUM_CLASSES)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = F.relu(self.input(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        x = self.softmax(self.output(x))

        return x




class DeepLayerModule(nn.Module):
    def __init__(self, in_features, out_features, N, LoRA: int = None, LOW_RANK_INIT:float = None):
        super(DeepLayerModule, self).__init__()
        if LoRA:
            self.layers = self.layers = nn.ModuleList(
                [LoRALinear(in_features, out_features, rank=max(int(LoRA * min(in_features, out_features)),1),alpha=1)] +  # W1 MxN
                [LoRALinear(out_features, out_features, rank=max(int(LoRA * min(in_features, out_features)),1),alpha=1) for _ in
                 range(N - 1)])  # W2...WN NxN
        else:
            self.layers = self.layers = nn.ModuleList([nn.Linear(in_features, out_features)] +  # W1 MxN
                                                      [nn.Linear(out_features, out_features) for _ in
                                                       range(N - 1)])  # W2...WN NxN

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x



class SimpleDNN(nn.Module):
    def __init__(self, layer_sizes=[512, 512, 512, 128], NUM_CLASSES=10,
                 IMAGE_SIZE=28, CHANNELS=1, dropout_prob=1.0):
        super(SimpleDNN, self).__init__()

        input_size = CHANNELS * IMAGE_SIZE ** 2
        self.layer_sizes = [input_size] + layer_sizes

        layers = []
        for i in range(len(self.layer_sizes) - 1):
            layers.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]))
            if i < len(self.layer_sizes) - 2:
                layers.append(nn.ReLU())

        # ✅ Wrap Sequential in a submodule called `model`
        self.model = nn.Sequential(*layers)
        self.output_layer = nn.Linear(self.layer_sizes[-1], NUM_CLASSES)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.model(x)
        x = self.output_layer(x)
        return x

class DNNWrapper(nn.Module):
    def __init__(self, dnn: nn.Module):
        super().__init__()
        self.model = dnn  #
        # This wraps SimpleDNN inside `model`

    def forward(self, x):
        return self.model(x)