import torch
from torch import nn


class ControllerV1(nn.Module):
    def __init__(self):
        super(ControllerV1, self).__init__()
        self.controller = nn.Sequential(
            nn.Linear(4 + 4 + 1, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 4),
            nn.Softplus()
        )

    def forward(self, x):
        x = self.controller(x)
        return torch.exp(x)
