from comet_ml import Experiment
import torch
from torch import nn
from torchdiffeq import odeint_adjoint as odeint
from double_pendulum.double_pendulum import DoublePendulumDiffEq
from double_pendulum.double_pendulum_approximation import DoublePendulumApproxDiffEq, \
    DoublePendulumApproxDiffEqCoordinates
from double_pendulum.generate_params import *  # TODO: remove *
from coordinate_utils import return_coordinates_double_pendulum
from vizualization_utils import plot_pendulums
import matplotlib.pyplot as plt
from controller_tuner_nets import ControllerV1 as Controller
from controller_tuner_nets import TunerAnglesV1 as Tuner
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
@click.option('--logging_period', type=int, default=10)
@click.option('--batch_size', type=int, default=200)
@click.option('--method', type=str, default='rk4')
@click.option('--external_force_1', type=str, default='lambda t: 0.')
@click.option('--external_force_2', type=str, default='lambda t: 0.')
@click.option('--controller_type', type=str, default='internal',
              help='None, internal or external_derivatives')
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
        controller_type: str,
):
    experiment = Experiment(project_name=project_name, workspace=work_space)
    experiment_key = experiment.get_key()
    PATH = './'
    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')

    ts = torch.arange(start=0, end=duration, step=step).float().to(device)
    double_pendulum = DoublePendulumDiffEq(
        external_force_1=external_force_1,
        external_force_2=external_force_2
    ).to(device)
    controller = Controller().to(device)
    tuner = Tuner(ar=5).to(device)

    loss_fn = nn.MSELoss()

    optimizer_controller = torch.optim.Adam(controller.parameters(), lr=lr, weight_decay=1e-4)
    optimizer_tuner = torch.optim.Adam(tuner.parameters(), lr=lr, weight_decay=1e-4)

    def check_coords(coord):
        # pass
        assert np.allclose((coord[0, :, 0] ** 2 + coord[1, :, 0] ** 2).detach().numpy(), 1), coord[:, 0, 0]

    # check_coords(coord_double_pend)
    # coord_double_pend = coord_double_pend + torch.randn_like(coord_double_pend) *noise  * coord_double_pend.std(
    #     dim=(0, 1))

    loss_best = 10000
    train_noise_std = .5  # TODO

    # training
    for epoch in range(epochs):
        # general
        print(epoch)
        train_inits = torch.clamp(torch.randn(batch_size, 4).float().to(device), -1, 1) / 2.
        coord_double_pend = torch.from_numpy(
            return_coordinates_double_pendulum(double_pendulum, train_inits, ts, noise=0))

        # controller
        controller.train(True)
        optimizer_controller.zero_grad()
        double_pendulum_approx_controller = DoublePendulumApproxDiffEq(
            controller=controller,
            controller_type=controller_type,
            init=train_inits,
            external_force_1=external_force_1,
            external_force_2=external_force_2
        ).to(device)

        # coord_double_pend_approx_controller = torch.from_numpy(return_coordinates_double_pendulum(
        #     double_pendulum_approx_controller, train_inits, ts, noise=0))

        coord = odeint(
            double_pendulum_approx_controller, train_inits, ts, rtol=1e-3, atol=1e-3, method=method)
        x1 = torch.sin(coord[:, :, 0])
        y1 = torch.cos(coord[:, :, 0])
        x2 = x1 + torch.sin(coord[:, :, 1])
        y2 = y1 + torch.cos(coord[:, :, 1])
        coord_double_pend_approx_controller = torch.stack([x1, y1, x2, y2])  # [coord, timestamp, batch]


        loss = loss_fn(coord_double_pend, coord_double_pend_approx_controller)
        loss.backward()
        optimizer_controller.step()
        experiment.log_metric('Train loss (DC)', loss.item(), step=epoch)
        controller.train(False)

        # tuner
        tuner.train(True)
        optimizer_tuner.zero_grad()
        double_pendulum_approx_tuner = DoublePendulumApproxDiffEqCoordinates(
            tuner=tuner,
            init=train_inits,
            external_force_1=external_force_1,
            external_force_2=external_force_2,
            method=method,
            ts=ts
        ).to(device)
        # double_pendulum_approx_tuner.reset(train_noise_std) # TODO: turn on
        # TODO: check transpose
        coord_double_pend_approx_tuner = torch.stack(
            [double_pendulum_approx_tuner(i) for i in range(len(ts))], 0).transpose(0, 1)
        loss = loss_fn(coord_double_pend, coord_double_pend_approx_tuner)
        loss.backward()
        optimizer_tuner.step()
        experiment.log_metric('Train loss (Tuner)', loss.item(), step=epoch)
        tuner.train(False)

        # general
        # train_noise_std *= 0.995
        # lr *= 0.999
        # for param_group in optimizer_controller.param_groups:
        #     param_group['lr'] = lr
        # for param_group in optimizer_tuner.param_groups:
        #     param_group['lr'] = lr
        #
        # save pics every 50 epochs
        if epoch % logging_period == 0:
            # test and plot
            with torch.no_grad():
                test_inits = torch.clamp(torch.randn(batch_size, 4).float().to(device), -1, 1) / 2.
                coord_double_pend = torch.from_numpy(
                    return_coordinates_double_pendulum(double_pendulum, test_inits, ts, noise=0))

                # controller
                controller.eval()
                controller.train(False)
                double_pendulum_approx_controller = DoublePendulumApproxDiffEq(
                    controller=controller,
                    controller_type=controller_type,
                    init=test_inits,
                    external_force_1=external_force_1,
                    external_force_2=external_force_2
                ).to(device)
                coord_double_pend_approx_controller = torch.from_numpy(return_coordinates_double_pendulum(
                    double_pendulum_approx_controller, test_inits, ts, noise=0))
                loss = loss_fn(coord_double_pend, coord_double_pend_approx_controller)
                experiment.log_metric('Test loss (DC)', loss.item(), step=epoch)

                # tuner
                tuner.eval()
                tuner.train(False)
                double_pendulum_approx_tuner = DoublePendulumApproxDiffEqCoordinates(
                    tuner=tuner,
                    init=test_inits,
                    external_force_1=external_force_1,
                    external_force_2=external_force_2,
                    method=method,
                    ts=ts
                ).to(device)
                coord_double_pend_approx_tuner = torch.stack(
                    [double_pendulum_approx_tuner(i) for i in range(len(ts))], 0)

                loss = loss_fn(coord_double_pend, coord_double_pend_approx_tuner.transpose(1, 0))
                experiment.log_metric('Test loss (Tuner)', loss.item(), step=epoch)

                # plot
                fig = plot_pendulums(d_true=coord_double_pend,
                                     d_dc=coord_double_pend_approx_controller + 0.02,
                                     d_approx=double_pendulum_approx_tuner.init_default_coords,
                                     d_tuner=coord_double_pend_approx_tuner.transpose(0, 1) + 0.04)
                experiment.log_figure(f"{epoch} Quality dynamic test", fig, step=epoch)
                plt.close()

            # saving weights
            # if loss_test.item() < loss_best:
            #     loss_best = loss_test.item()
            #     print(loss_best, end=' ')

            #     best_weights = copy.deepcopy(tuner.state_dict())
            #     torch.save(best_weights, open(PATH + 'tuner_{}.pcl'.format(experiment_key), 'wb+'))


if __name__ == '__main__':
    main()
