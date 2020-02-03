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
        # p = 0.2
        # self.controller = nn.Sequential(
        #     nn.Linear(4, 16),
        #     nn.Dropout(p),
        #     nn.Tanh(),
        #     nn.Linear(16, 16),
        #     nn.Dropout(p),
        #     nn.Tanh(),
        #     nn.Linear(16, 16),
        #     nn.Dropout(p),
        #     nn.Tanh(),
        #     nn.Linear(16, 4),
        #     nn.Tanh()
        # )

    def forward(self, x):
        return self.controller(x) * 50  # TODO: limit values with less crutches


class TunerCoordinatesV1(nn.Module):
    def __init__(self):
        # np.stack([x1, y1, x2, y2])  # [coord, timestamp, batch]
        super(TunerCoordinatesV1, self).__init__()
        self.tuner = nn.Sequential(
            nn.Linear(4, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 4),
            nn.Tanh()
        )

    def forward(self, x):
        return self.tuner(x) * 2


class TunerAnglesV1(TunerCoordinatesV1):
    def __init__(self):
        super(TunerAnglesV1, self).__init__()

    # TODO: add length dependency
    # TODO: пофиксить резкий переход в 0 на 360
    def _coordinates2angles(coords):
        angles = np.stack([
            np.angle(coords[0] + 1j * coords[1]),
            np.angle(coords[2] + 1j * coords[3])
        ])
        angles += np.pi / 2
        return angles

    def _angles2coordinates(angles):
        angles -= np.pi / 2
        x1 = np.cos(angles[0])
        y1 = np.sin(angles[0])
        x2 = np.cos(angles[1])
        y2 = np.sin(angles[1])
        #     print(angles[(angles > -2) * (angles < -1)])
        #     print(angles)
        return np.stack([x1, y1, x2, y2]).round(decimals=5)

    def forward(self, x):
        x = self._coordinates2angles(x)
        pred = self.tuner(x) * 180
        pred = self._angles2coordinates(pred)
        return pred
