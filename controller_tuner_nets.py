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
    def __init__(self, ar=3):
        # np.stack([x1, y1, x2, y2])  # [coord, timestamp, batch]
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
    def __init__(self, ar=3):
        super(TunerAnglesV1, self).__init__()
        self.ar = ar
        self.tuner = nn.Sequential(
            nn.Linear(2 * ar, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 2),
            nn.Tanh()
        )

    # TODO: add length dependency
    # TODO: пофиксить резкий переход в 0 на 360
    # TODO: faster using torch stuff instead of numpy?
    @staticmethod
    def _coordinates2angles(coords, is_print=False):
        coords = coords.numpy()
        if is_print:
            print("coord2ang", coords, coords.shape)
        angles = np.stack([
            np.angle(coords[:, 0] + 1j * coords[:, 1]),
            np.angle(coords[:, 2] + 1j * coords[:, 3])
        ]).T
        if is_print:
            coords = coords.reshape(-1)
            temp_angles = np.stack([
                np.angle(coords[0] + 1j * coords[1]),
                np.angle(coords[2] + 1j * coords[3])
            ])
            print("temp_angles", temp_angles)
        return torch.from_numpy(angles)

    def _coordinates2angles_ar(self, coords):
        angles = []
        for i in range(self.ar):
            ang = self._coordinates2angles(coords[:, i * 4:(i + 1) * 4])
            angles.append(ang)
        angles = np.hstack(angles)
        return torch.from_numpy(angles)

    @staticmethod
    def _angles2coordinates(angles):
        x1 = torch.cos(angles[:, 0])
        y1 = torch.sin(angles[:, 0])
        x2 = torch.cos(angles[:, 1])  # * 2
        y2 = torch.sin(angles[:, 1])  # * 2
        return torch.stack([x1, y1, x2, y2])

    def forward(self, x):
        print("input", x.view(self.ar, -1, 4)[-1, 0, :], x.shape)
        angles = self._coordinates2angles_ar(x)
        temp_angles = self._coordinates2angles(x.view(self.ar, -1, 4)[-1, 0, :].view(-1, 4), is_print=True)
        print("angles", temp_angles, temp_angles.shape)
        temp_coodrs = self._angles2coordinates(temp_angles)
        print("xxx3.5", temp_coodrs.view(-1, 4)[0, :], temp_coodrs.shape)
        pred = temp_angles + (self.tuner(angles + np.pi / 2) - np.pi / 2) * 0.001
        pred = self._angles2coordinates(pred)
        # TODO: проверить что координаты правильные вообще
        print("xxx4", pred.view(-1, 4)[-1, 0], x.shape)
        # return x.view(self.ar, -1, 4)[-1] + 0.1 * pred.view(-1, 4)
        return pred.view(-1, 4)
