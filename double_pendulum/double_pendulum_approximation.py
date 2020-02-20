import pyro
import torch
from torch import nn
from torchdiffeq import odeint_adjoint as odeint
import numpy as np


class DoublePendulumApproxDiffEq(nn.Module):
    def __init__(self,
                 init,
                 controller=None,
                 controller_type=None,
                 mass_1=1.,
                 mass_2=1.,
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
                 ts,
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

        self.ts = ts
        self._double_pendulum_approx = DoublePendulumApproxDiffEq(
            controller=None,
            init=init,
            external_force_1=external_force_1,
            external_force_2=external_force_2,
        )  # .to(torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
        # TODO: make it faster
        # TODO: ? import
        coord = odeint(self._double_pendulum_approx, self.init, ts, rtol=1e-3, atol=1e-3, method=method)
        x1 = np.sin(coord[:, :, 0].detach().cpu().numpy())
        y1 = np.cos(coord[:, :, 0].detach().cpu().numpy())
        x2 = x1 + np.sin(coord[:, :, 1].detach().cpu().numpy())
        y2 = y1 + np.cos(coord[:, :, 1].detach().cpu().numpy())
        self._default_coords = np.stack([x1, y1, x2, y2])  # [4, len(ts), batch_size]
        self._init_default_coords = self._default_coords.copy()

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

    def forward(self, n_t):
        if n_t <= self.tuner.ar:
            return torch.from_numpy(self._default_coords[:, n_t, :])
        else:
            coordinates = torch.from_numpy(self._default_coords[:, n_t - self.tuner.ar + 1:n_t + 1, :])
            pred_coordinates = self.tuner(
                coordinates)  # .to(torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
            self._default_coords[:, n_t, :] = pred_coordinates.detach().numpy()
            return pred_coordinates

    def reset(self, noise_std=0.):
        self._default_coords = self._init_default_coords.copy()
        noise = np.random.normal(0, noise_std, self._default_coords.shape)
        self._default_coords += noise
