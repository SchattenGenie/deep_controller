import torch
from torch import nn
from torchdiffeq import odeint_adjoint as odeint


class DoublePendulumDiffEq(nn.Module):
    def __init__(self,
                 mass_1=1.,
                 mass_2=1.,
                 length_1=1.,
                 length_2=1.,
                 damping_1=-0.01,
                 damping_2=-0.01,
                 external_force_1=lambda t: 0.,
                 external_force_2=lambda t: 0.,
                 g=9.8
                 ):
        super().__init__()
        self.mass_1 = mass_1
        self.mass_2 = mass_2
        self.length_1 = length_1
        self.length_2 = length_2
        self.damping_1 = damping_1
        self.damping_2 = damping_2
        self.external_force_1 = external_force_1
        self.external_force_2 = external_force_2
        self.noise = 1.
        self.g = g

    def forward(self, t, x):
        d_theta_1 = x[:, 2]
        d_theta_2 = x[:, 3]
        theta_1 = x[:, 0]
        theta_2 = x[:, 1]
        delta_theta = theta_2 - theta_1
        M = self.mass_1 + self.mass_2

        right_part_1 = self.mass_2 * self.length_2 * d_theta_2.pow(2) * delta_theta.sin() - M * self.g * theta_1.sin() + self.damping_1 * d_theta_1 + self.external_force_1(t)
        denominator_1 = self.length_1 * (M - self.mass_2 * delta_theta.cos().pow(2))
        mul_1 = self.mass_2 * delta_theta.cos()

        right_part_2 = - self.length_1 * d_theta_1.pow(2) * delta_theta.sin() - self.g * theta_2.sin() + self.damping_2 * d_theta_2 + self.external_force_2(t)
        denominator_2 = self.length_2 * (1 - self.mass_2 * delta_theta.cos().pow(2) / M)
        mul_2 = delta_theta.cos() / M

        d_phi_1 = (right_part_1 - right_part_2 * mul_1) / denominator_1
        d_phi_2 = (right_part_2 - right_part_1 * mul_2) / denominator_2
        return torch.stack([d_theta_1, d_theta_2, d_phi_1, d_phi_2]).t()
