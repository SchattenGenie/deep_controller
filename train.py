from comet_ml import Experiment
import torch
from torch import nn
import numpy as np
from torchdiffeq import odeint_adjoint as odeint
from double_pendulum.double_pendulum import DoublePendulumDiffEq
from double_pendulum.double_pendulum_approximation import DoublePendulumApproxDiffEq
import matplotlib.pyplot as plt
from tqdm import tqdm
import click
import copy


# TODO: get corresponding smth
def return_coordinates_double_pendulum(double_pendulum, inits, ts, method='rk4'):
    coord = odeint(double_pendulum, inits, ts, rtol=1e-3, atol=1e-3, method=method)
    x1 = np.sin(coord[:, :, 0].detach().cpu().numpy())
    y1 = np.cos(coord[:, :, 0].detach().cpu().numpy())
    x2 = x1 + np.sin(coord[:, :, 1].detach().cpu().numpy())
    y2 = y1 + np.cos(coord[:, :, 1].detach().cpu().numpy())
    return np.stack([x1, y1, x2, y2])  # [coord, timestamp, batch]


def plot_pendulums(d1, d2, component=0):
    fig, ax = plt.subplots(2, 2, figsize=(12, 12), dpi=200)
    titles = ["x1", "y1", "x2", "y2"]
    x = np.arange(d1.shape[1])
    for i in range(2):
        for j in range(2):
            ax[i][j].plot(x, d1[i * 2 + j, :, component], label='True')
            ax[i][j].plot(x, d2[i * 2 + j, :, component], label='Approx')
            ax[i][j].set_title(titles[i * 2 + j])
            ax[i][j].legend()
    return fig


@click.command()
@click.option('--project_name', type=str, prompt='Enter project name')
@click.option('--work_space', type=str, prompt='Enter workspace name')
@click.option('--epochs', type=int, default=50000)
@click.option('--lr', type=float, default=2e-4)
@click.option('--method', type=str, default='rk4')
def main(project_name, work_space, epochs, lr, method):
    experiment = Experiment(project_name=project_name, workspace=work_space)
    experiment_key = experiment.get_key()
    PATH = './'
    device = torch.device('cuda:2')

    ts = torch.arange(start=0, end=4, step=0.05).float().to(device)
    train_inits = torch.clamp(torch.randn(20, 4).float().to(device), -1, 1) / 2.
    test_inits = torch.clamp(torch.randn(20, 4).float().to(device), -1, 1) / 2.

    controller = nn.Sequential(
        nn.Linear(4 + 4 + 1, 16),
        nn.Tanh(),
        nn.Linear(16, 16),
        nn.Tanh(),
        nn.Linear(16, 16),
        nn.Tanh(),
        nn.Linear(16, 4),
        nn.Tanh()
    ).to(device)

    double_pendulum = DoublePendulumDiffEq().to(device)
    double_pendulum_approx = DoublePendulumApproxDiffEq(controller=controller, init=train_inits).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(controller.parameters(), lr=lr, weight_decay=1e-4)

    coord_double_pend = odeint(double_pendulum, train_inits, ts, rtol=1e-3, atol=1e-3, method=method).detach().clone()
    best_weights = copy.deepcopy(controller.state_dict())
    loss_best = 10000
    for epoch in range(epochs):
        optimizer.zero_grad()
        coord_double_pend_approx = odeint(double_pendulum_approx, train_inits, ts, rtol=1e-3, atol=1e-3, method=method)
        loss = loss_fn(coord_double_pend, coord_double_pend_approx)
        if loss.item() < loss_best:
            loss_best = loss.item()
            best_weights = copy.deepcopy(controller.state_dict())
            torch.save(best_weights, open(PATH + 'controller_{}.pcl'.format(experiment_key), 'wb+'))
        loss.backward()
        optimizer.step()

        experiment.log_metric('Train loss', loss.item(), step=epoch)
        if epoch % 5 == 0:
            with torch.no_grad():
                data_pendulum_approx = return_coordinates_double_pendulum(double_pendulum_approx, test_inits, ts)
                data_pendulum = return_coordinates_double_pendulum(double_pendulum, test_inits, ts)
                loss_test = np.sqrt(((data_pendulum_approx - data_pendulum)**2).mean())
            experiment.log_metric('Test loss', loss_test, step=epoch)
            fig = plot_pendulums(data_pendulum_approx, data_pendulum)
            experiment.log_figure("Quality dynamic test", fig, step=epoch)
            plt.close()

            with torch.no_grad():
                data_pendulum_approx = return_coordinates_double_pendulum(double_pendulum_approx, train_inits, ts)
                data_pendulum = return_coordinates_double_pendulum(double_pendulum, train_inits, ts)
            fig = plot_pendulums(data_pendulum_approx, data_pendulum)
            experiment.log_figure("Quality dynamic train", fig, step=epoch)
            plt.close()


if __name__ == '__main__':
    main()
