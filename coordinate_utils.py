import numpy as np
import torch
from torchdiffeq import odeint_adjoint as odeint
from double_pendulum.double_pendulum import DoublePendulumDiffEq
from double_pendulum.double_pendulum_approximation import DoublePendulumApproxDiffEq


# TODO: use mass, length from controller
def return_coordinates_double_pendulum(double_pendulum, inits, ts, method='rk4', noise=0.):
    coord = odeint(double_pendulum, inits, ts, rtol=1e-3, atol=1e-3, method=method)
    coord = coord + torch.randn_like(coord) * noise * coord.std(dim=(0, 1))
    x1 = np.sin(coord[:, :, 0].detach().cpu().numpy())
    y1 = np.cos(coord[:, :, 0].detach().cpu().numpy())
    x2 = x1 + np.sin(coord[:, :, 1].detach().cpu().numpy())
    y2 = y1 + np.cos(coord[:, :, 1].detach().cpu().numpy())
    return np.stack([x1, y1, x2, y2])  # [coord, timestamp, batch]


# TODO: get corresponding smth
def return_coordinates_double_pendulum_controller(double_pendulum, inits, ts, method='rk4', noise=0.):
    # [timestamp, batch, coord]
    coord = odeint(double_pendulum, inits, ts, rtol=1e-3, atol=1e-3, method=method)

    if isinstance(double_pendulum, DoublePendulumApproxDiffEq) and double_pendulum.controller is not None:
        # [timestamp x batch, coord]
        coord_controller = coord.view(-1, 4)
        ts_controller = ts.view(-1, 1).repeat(1, coord.shape[1]).view(-1, 1)
        inits_controller = inits.view(1, inits.shape[0], 4).repeat(coord.shape[0], 1, 1).view(-1, 4)
        data_controller = torch.cat([coord_controller, ts_controller, inits_controller], dim=1)
        data = double_pendulum.controller(data_controller)
        # [(mass_1, mass_2, length_1, length_2), timestamp, batch]
        data = data.view(4, len(ts), coord.shape[1]).detach().cpu().numpy()
    else:
        data = np.array([
            double_pendulum.mass_1,
            double_pendulum.mass_2,
            double_pendulum.length_1,
            double_pendulum.length_1
        ])
        data = np.transpose(np.tile(data, [coord.shape[1], len(ts), 1]), [2, 1, 0])
    coord = coord.detach().cpu().numpy()
    # length_1 * sin(theta_1)
    x1 = data[2, :, :] * np.sin(coord[:, :, 0])
    # length_1 * cos(theta_1)
    y1 = data[2, :, :] * np.cos(coord[:, :, 0])
    # # x1 + length_2 * sin(theta_2)
    x2 = x1 + data[3, :, :] * np.sin(coord[:, :, 1])
    # # y1 + length_2 * cos(theta_2)
    y2 = y1 + data[3, :, :] * np.cos(coord[:, :, 1])

    return np.stack([x1, y1, x2, y2])  # [coord, timestamp, batch]
