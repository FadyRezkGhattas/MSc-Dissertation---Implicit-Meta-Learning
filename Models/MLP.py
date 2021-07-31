import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=784, output_dim=10, nhid = 32):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, nhid),
            nn.LeakyReLU(),
            nn.Linear(nhid, output_dim)
        )

    def forward(self, x):
        return self.layers(x)