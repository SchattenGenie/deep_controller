import pyro
import torch
from torch import nn
from torchdiffeq import odeint
import numpy as np


class DoublePendulumApproxDiffEq(nn.Module):
    def __init__(self,
                 init,
                 controller=None,
                 controller_type=None,
                 mass_1=1., mass_2=1.,
                 length_1=1.,
                 length_2=1.,
                 damping_1=-0.01,
                 damping_2=-0.01,
                 external_force_1="lambda t: 0.",
                 external_force_2="lambda t: 0.",
                 g=9.8):
        super().__init__()
        self.mass_1 = mass_1
        self.mass_2 = mass_2
        self.length_1 = length_1
        self.length_2 = length_2
        self.controller = controller
        self.controller_type = controller_type
        self.init = init
        self.damping_1 = damping_1
        self.damping_2 = damping_2
        self.external_force_1 = eval(external_force_1)
        self.external_force_2 = eval(external_force_2)
        self.noise = 1.
        self.g = g

    def forward(self, t, x):
        print("t:", t)
        if self.controller is None or self.controller_type != "internal":
            mass_1, mass_2, length_1, length_2 = self.mass_1, self.mass_2, self.length_1, self.length_2
        else:
            inp = x.view(-1, 4)
            inp = torch.cat([inp, t.repeat(len(x), 1), self.init], dim=1)
            pred = self.controller(inp)  # previous and init
            mass_1, mass_2, length_1, length_2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]

        d_theta_1 = x[:, 2]
        d_theta_2 = x[:, 3]
        theta_1 = x[:, 0]
        theta_2 = x[:, 1]
        delta_theta = theta_2 - theta_1
        M = mass_1 + mass_2

        right_part_1 = mass_2 * length_2 * d_theta_2.pow(
            2) * delta_theta - M * self.g * theta_1 + self.damping_1 * d_theta_1 + self.external_force_1(t)
        denominator_1 = length_1 * (M - mass_2)
        mul_1 = mass_2

        right_part_2 = - length_1 * d_theta_1.pow(
            2) * delta_theta - self.g * theta_2 + self.damping_2 * d_theta_2 + self.external_force_2(t)
        denominator_2 = length_2 * (1 - mass_2 / M)
        mul_2 = 1. / M

        d_phi_1 = (right_part_1 - right_part_2 * mul_1) / denominator_1
        d_phi_2 = (right_part_2 - right_part_1 * mul_2) / denominator_2

        if self.controller is not None and self.controller_type == "external_derivatives":
            inp = torch.stack([d_theta_1, d_theta_2, d_phi_1, d_phi_2]).t()
            pred = self.controller(inp)
            d_theta_1, d_theta_2, d_phi_1, d_phi_2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]

        return torch.stack([d_theta_1, d_theta_2, d_phi_1, d_phi_2]).t()


# TODO: inheritance
class DoublePendulumApproxDiffEqCoordinates(nn.Module):
    def __init__(self,
                 init,
                 tuner,
                 mass_1=1., mass_2=1.,
                 length_1=1.,
                 length_2=1.,
                 damping_1=-0.01,
                 damping_2=-0.01,
                 external_force_1="lambda t: 0.",
                 external_force_2="lambda t: 0.",
                 method='rk4',
                 g=9.8):
        super().__init__()
        self.mass_1 = mass_1
        self.mass_2 = mass_2
        self.length_1 = length_1
        self.length_2 = length_2
        self.tuner = tuner
        self.init = init
        self.damping_1 = damping_1
        self.damping_2 = damping_2
        self.external_force_1 = eval(external_force_1)
        self.external_force_2 = eval(external_force_2)
        self.noise = 1.
        self.g = g
        self.method = method

    def _derivatives(self, t, x):
        d_theta_1 = x[:, 2]
        d_theta_2 = x[:, 3]
        theta_1 = x[:, 0]
        theta_2 = x[:, 1]
        delta_theta = theta_2 - theta_1
        M = self.mass_1 + self.mass_2

        right_part_1 = self.mass_2 * self.length_2 * d_theta_2.pow(
            2) * delta_theta - M * self.g * theta_1 + self.damping_1 * d_theta_1 + self.external_force_1(t)
        denominator_1 = self.length_1 * (M - self.mass_2)
        mul_1 = self.mass_2

        right_part_2 = - self.length_1 * d_theta_1.pow(
            2) * delta_theta - self.g * theta_2 + self.damping_2 * d_theta_2 + self.external_force_2(t)
        denominator_2 = self.length_2 * (1 - self.mass_2 / M)
        mul_2 = 1. / M

        d_phi_1 = (right_part_1 - right_part_2 * mul_1) / denominator_1
        d_phi_2 = (right_part_2 - right_part_1 * mul_2) / denominator_2

        return torch.stack([d_theta_1, d_theta_2, d_phi_1, d_phi_2]).t()

    # TODO: x -- это и есть координаты видимо
    def forward(self, t):
        coodinates = odeint(self._derivatives, self.init, torch.from_numpy(np.array([t])), rtol=1e-3, atol=1e-3, method=self.method).detach().clone()
        # TODO: change self.init?
        inp = coodinates.view(-1, 4)
        pred_coordinates = self.tuner(inp)  # previous and init
        return pred_coordinates
