import pyro
import torch
from torch import nn
from torchdiffeq import odeint_adjoint as odeint


class DoublePendulumApproxDiffEq(nn.Module):
    def __init__(self, init, controller=None, mass_1=1., mass_2=1., length_1=1., length_2=1., g=9.8):
        super().__init__()
        self.mass_1 = mass_1
        self.mass_2 = mass_2
        self.length_1 = length_1
        self.length_2 = length_2
        self.controller = controller
        self.init = init
        self.g = g

    def forward(self, t, x):
        d_theta_1 = x[:, 2]
        d_theta_2 = x[:, 3]
        if self.controller is None:
            mass_1, mass_2, length_1, length_2 = self.mass_1, self.mass_2, self.length_1, self.length_2
        else:
            inp = x.view(-1, 4)
            inp = torch.cat([inp, t.repeat(len(x), 1), self.init], dim=1)
            pred = 2. * (1.01 + self.controller(inp))  # previous and init
            mass_1, mass_2, length_1, length_2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]

        M = self.mass_1 + self.mass_2
        d_phi_1 = (
                          - self.g * (M + mass_1) * x[:, 0]
                          - mass_2 * self.g * (x[:, 0] - 2 * x[:, 1])
                          - 2 * (x[:, 0] - x[:, 1]) * mass_2 * (x[:, 2].pow(2) * length_1 + x[:, 3].pow(2) * length_2)
                  ) / (
                          length_1 * (M + mass_1 - mass_2)
                  )
        d_phi_2 = (
                2 * (x[:, 0] - x[:, 1]) * (
                M * x[:, 2] ** 2 * length_1 + self.g * M + x[:, 3].pow(2) * length_2 * mass_2
        ) / (
                        length_2 * (M + mass_1 - mass_2)
                )
        )
        return torch.stack([d_theta_1, d_theta_2, d_phi_1, d_phi_2]).t()