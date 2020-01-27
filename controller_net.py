import torch
from torch import nn


class ControllerV1(nn.Module):
    def __init__(self, controller_type="internal"):
        super(ControllerV1, self).__init__()
        if controller_type == "internal":
            n_input = 4 + 4 + 1
        elif controller_type == "external":
            n_input = 4
        if controller_type not in ["internal", "external"]:
            raise ValueError("controller_type should be internal or external")
        self.controller = nn.Sequential(
            nn.Linear(n_input, 16),
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
