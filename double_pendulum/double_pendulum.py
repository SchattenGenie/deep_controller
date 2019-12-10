import torch
from torch import nn
from torchdiffeq import odeint_adjoint as odeint


class DoublePendulumDiffEq(nn.Module):
    def __init__(self, mass_1=1., mass_2=1., length_1=1., length_2=1., g=9.8):
        super().__init__()
        self.mass_1 = mass_1
        self.mass_2 = mass_2
        self.length_1 = length_1
        self.length_2 = length_2
        self.g = g

    def forward(self, t, x):
        d_theta_1 = x[:, 2]
        d_theta_2 = x[:, 3]
        M = self.mass_1 + self.mass_2
        d_phi_1 = (
                          - self.g * (M + self.mass_1) * x[:, 0].sin()
                          - self.mass_2 * self.g * (x[:, 0] - 2 * x[:, 1]).sin()
                          - 2 * (x[:, 0] - x[:, 1]).sin() * self.mass_2 * (
                                      x[:, 2].pow(2) * self.length_1 * (x[:, 0] - x[:, 1]).cos() + x[:, 3].pow(
                                  2) * self.length_2)
                  ) / (
                          self.length_1 * (M + self.mass_1 - self.mass_2 * (2 * x[:, 0] - 2 * x[:, 1]).cos())
                  )
        d_phi_2 = (
                2 * (x[:, 0] - x[:, 1]).sin() * (
                M * x[:, 2] ** 2 * self.length_1 + self.g * M * x[:, 0].cos() + x[:, 3].pow(
            2) * self.length_2 * self.mass_2 * (x[:, 0] - x[:, 1]).cos()
        ) / (
                        self.length_2 * (M + self.mass_1 - self.mass_2 * (2 * x[:, 0] - 2 * x[:, 1]).cos())
                )
        )
        return torch.stack([d_theta_1, d_theta_2, d_phi_1, d_phi_2]).t()