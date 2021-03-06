import torch
from torch import nn
import numpy as np


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


class ControllerExternalDerivativesV1(ControllerV1):
    # TODO: universal Controller for all external controllers
    def __init__(self):
        super().__init__()
        self.controller[0] = nn.Linear(4, 16)
        self.controller[-1] = nn.Tanh()
        # TODO: remove Dropout when apply -- add controller.eval() everywhere
        # TODO: check

    def forward(self, x):
        return self.controller(x) * 50  # TODO: limit values with less crutches


class TunerCoordinatesV1(nn.Module):
    def __init__(self, ar=3):
        super(TunerCoordinatesV1, self).__init__()
        self.ar = ar
        self.tuner = nn.Sequential(
            nn.Linear(4 * ar, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 4),
            nn.Tanh()
        )

    def forward(self, x):
        # TODO: x2, y2 += 1
        return self.tuner(x) * 2


class TunerAnglesV1(nn.Module):
    def __init__(self, ar=3, dropout=0.1):
        super(TunerAnglesV1, self).__init__()
        self.ar = ar
        self.tuner = nn.Sequential(
            nn.Linear(2 * ar, 16),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(16, 2),
            nn.Tanh()
        )
        # self._i = 0

    # TODO: add length dependency
    # TODO: пофиксить резкий переход в 0 на 360
    # TODO: faster using torch stuff instead of numpy?
    @staticmethod
    def _coordinates2angles(coords):
        coords = coords.cpu().numpy()
        angles = np.stack([
            np.angle(coords[0] + 1j * coords[1]),
            np.angle(coords[2] - coords[0] + 1j * (coords[3] - coords[1]))
        ]).T
        return angles

    def _coordinates2angles_ar(self, coords):
        angles = []
        for i in range(self.ar):
            ar_coords = coords[:, i]
            ang = self._coordinates2angles(ar_coords)
            angles.append(ang)
        angles = np.stack(angles)
        angles = angles.transpose((2, 0, 1))

        return angles

    @staticmethod
    def _angles2coordinates(angles):
        x1 = torch.cos(angles[0])
        y1 = torch.sin(angles[0])
        x2 = torch.cos(angles[1]) + x1
        y2 = torch.sin(angles[1]) + y1
        return torch.stack([x1, y1, x2, y2])

    def forward(self, x, a=1.):
        angles = self._coordinates2angles_ar(x)
        pred_angles = angles.reshape(-1, self.ar * 2)
        pred_angles = torch.from_numpy(pred_angles)
        pred_angles = self.tuner(pred_angles) * 2 * np.pi
        pred_angles = torch.from_numpy(angles[:, -1, :]) + a * pred_angles.T
        # pred_angles = (1 - a) * torch.from_numpy(angles[:, -1, :]) + a * pred_angles.T
        # pred_angles = pred_angles.T
        pred = self._angles2coordinates(pred_angles)
        return pred
