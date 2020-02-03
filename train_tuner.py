from comet_ml import Experiment
import torch
from torch import nn
from torchdiffeq import odeint_adjoint as odeint
from double_pendulum.double_pendulum import DoublePendulumDiffEq
from double_pendulum.double_pendulum_approximation import DoublePendulumApproxDiffEqCoordinates
from coordinate_utils import return_coordinates_double_pendulum
from vizualization_utils import plot_pendulums
import matplotlib.pyplot as plt
from controller_tuner_nets import TunerCoordinatesV1, TunerAnglesV1
from tqdm import tqdm
import numpy as np
import click
import copy


# TODO: add batch size train / test
# TODO: controller configurable
# TODO: rtol/etol?
@click.command()
@click.option('--project_name', type=str, prompt='Enter project name')
@click.option('--work_space', type=str, prompt='Enter workspace name')
@click.option('--epochs', type=int, default=50000)
@click.option('--lr', type=float, default=2e-4)
@click.option('--duration', type=float, default=8)
@click.option('--step', type=float, default=0.05)
@click.option('--noise', type=float, default=1e-3)
@click.option('--logging_period', type=int, default=100)
@click.option('--batch_size', type=int, default=200)
@click.option('--method', type=str, default='rk4')
@click.option('--external_force_1', type=str, default='lambda t: 0.')
@click.option('--external_force_2', type=str, default='lambda t: 0.')
def main(
        project_name: str,
        work_space: str,
        epochs: int,
        lr: float,
        noise: float,
        step: float,
        duration: float,
        logging_period: int,
        external_force_1: str,
        external_force_2: str,
        method: str,
        batch_size: int,
):
    experiment = Experiment(project_name=project_name, workspace=work_space)
    experiment_key = experiment.get_key()
    PATH = './'
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    ts = torch.arange(start=0, end=duration, step=step).float().to(device)

    train_inits = torch.clamp(torch.randn(batch_size, 4).float().to(device), -1, 1) / 2.
    test_inits = torch.clamp(torch.randn(batch_size, 4).float().to(device), -1, 1) / 2.
    # TODO: mass, length, etc initializations in batch fashion
    # TODO: customization(i.e. several phase inits per mass init, several mass inits per phase init, force, etc

    tuner = TunerCoordinatesV1().to(device)

    double_pendulum = DoublePendulumDiffEq(
        external_force_1=external_force_1,
        external_force_2=external_force_2
    ).to(device)
    double_pendulum_approx = DoublePendulumApproxDiffEqCoordinates(
        tuner=tuner,
        init=train_inits,
        external_force_1=external_force_1,
        external_force_2=external_force_2,
        method=method
    ).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(tuner.parameters(), lr=lr, weight_decay=1e-4)
    best_weights = copy.deepcopy(tuner.state_dict())

    coord_double_pend = odeint(double_pendulum, train_inits, ts, rtol=1e-3, atol=1e-3, method=method).detach().clone()
    coord_double_pend = coord_double_pend + torch.randn_like(coord_double_pend) * noise * coord_double_pend.std(
        dim=(0, 1))
    loss_best = 10000
    for epoch in range(epochs):
        optimizer.zero_grad()
        # coord_double_pend_approx = odeint(double_pendulum_approx, train_inits, ts, rtol=1e-3, atol=1e-3, method=method)
        # coord_double_pend_approx = torch.cat([double_pendulum_approx(t) for t in ts]).view(-1, batch_size, 4) # TODO
        coord_double_pend_approx = []
        is_first = True
        coord_prev = None
        for t in ts:
            if is_first:
                coord = double_pendulum_approx(t)
                coord_double_pend_approx.append(coord)
                coord_prev = coord
                is_first = False
            else:
                coord = double_pendulum_approx(t, coord_prev)
                coord_double_pend_approx.append(coord)
                coord_prev = coord
        coord_double_pend_approx = torch.cat(coord_double_pend_approx).view(-1, batch_size, 4)
        loss = loss_fn(coord_double_pend, coord_double_pend_approx)
        loss.backward()
        optimizer.step()

        # testing
        with torch.no_grad():
            double_pendulum_approx_test = DoublePendulumApproxDiffEqCoordinates(
                tuner=tuner,
                init=test_inits,
                external_force_1=external_force_1,
                external_force_2=external_force_2,
                method=method
            ).to(device)
            # data_pendulum_approx = return_coordinates_double_pendulum(double_pendulum_approx_test, test_inits, ts,
            #                                                           noise=0.)
            # data_pendulum = return_coordinates_double_pendulum(double_pendulum, test_inits, ts, noise=noise)
            # TODO: discuss
            coord_double_pend = odeint(double_pendulum, test_inits, ts, rtol=1e-3, atol=1e-3,
                                       method=method).detach().clone()
            coord_double_pend = coord_double_pend + torch.randn_like(coord_double_pend) * noise * coord_double_pend.std(
                dim=(0, 1))
            # coord_double_pend_approx = torch.cat([double_pendulum_approx_test(t) for t in ts]).view(-1, batch_size, 4) # TODO
            coord_double_pend_approx = []
            is_first = True
            coord_prev = None
            for t in ts:
                if is_first:
                    coord = double_pendulum_approx_test(t)
                    coord_double_pend_approx.append(coord)
                    coord_prev = coord
                    is_first = False
                else:
                    coord = double_pendulum_approx_test(t, coord_prev)
                    coord_double_pend_approx.append(coord)
                    coord_prev = coord
            coord_double_pend_approx = torch.cat(coord_double_pend_approx).view(-1, batch_size, 4)
            loss_test = loss_fn(coord_double_pend, coord_double_pend_approx)

        # saving weights
        if loss_test.item() < loss_best:
            loss_best = loss_test.item()
            print(loss_best, end=' ')

            best_weights = copy.deepcopy(tuner.state_dict())
            torch.save(best_weights, open(PATH + 'tuner_{}.pcl'.format(experiment_key), 'wb+'))

        experiment.log_metric('Train loss', loss.item(), step=epoch)
        experiment.log_metric('Test loss', loss_test, step=epoch)

        # save pics every 50 epochs
        if epoch % logging_period == 0:
            with torch.no_grad():
                # data_pendulum_approx = return_coordinates_double_pendulum(double_pendulum_approx_test, test_inits, ts,
                #                                                           noise=0.)

                # data_pendulum = return_coordinates_double_pendulum(double_pendulum, test_inits, ts, noise=noise)
                log_coord_double_pend = return_coordinates_double_pendulum(double_pendulum, test_inits, ts, noise=noise)
                # coord_double_pend_approx = torch.cat([double_pendulum_approx(t) for t in ts]).view(-1, batch_size, 4)
                coord_double_pend_approx = []
                is_first = True
                coord_prev = None
                for t in ts:
                    if is_first:
                        coord = double_pendulum_approx_test(t)
                        coord_double_pend_approx.append(coord)
                        coord_prev = coord
                        is_first = False
                    else:
                        coord = double_pendulum_approx_test(t, coord_prev)
                        coord_double_pend_approx.append(coord)
                        coord_prev = coord
                coord_double_pend_approx = torch.cat(coord_double_pend_approx).view(4, -1, batch_size) #TODO: check reshape
                fig = plot_pendulums(log_coord_double_pend, coord_double_pend_approx)
                experiment.log_figure("Quality dynamic test", fig, step=epoch)
                plt.close()

                log_coord_double_pend = return_coordinates_double_pendulum(double_pendulum, train_inits, ts, noise=noise)
                coord_double_pend_approx = []
                is_first = True
                coord_prev = None
                for t in ts:
                    if is_first:
                        coord = double_pendulum_approx(t)
                        coord_double_pend_approx.append(coord)
                        coord_prev = coord
                        is_first = False
                    else:
                        coord = double_pendulum_approx(t, coord_prev)
                        coord_double_pend_approx.append(coord)
                        coord_prev = coord
                coord_double_pend_approx = torch.cat(coord_double_pend_approx).view(4, -1, batch_size) #TODO: check reshape
                fig = plot_pendulums(log_coord_double_pend, coord_double_pend_approx)
                experiment.log_figure("Quality dynamic train", fig, step=epoch)
                plt.close()


# TODO: remove duplicates (loops)
if __name__ == '__main__':
    main()
