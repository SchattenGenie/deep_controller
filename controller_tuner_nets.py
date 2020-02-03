import torch
from torch import nn


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
