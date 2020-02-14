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
# from controller_tuner_nets import TunerCoordinatesV1 as Tuner
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
@click.option('--logging_period', type=int, default=50)
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
    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')

    ts = torch.arange(start=0, end=duration, step=step).float().to(device)
    double_pendulum = DoublePendulumDiffEq(
        external_force_1=external_force_1,
        external_force_2=external_force_2
    ).to(device)
    tuner = Tuner(ar=20).to(device)

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(tuner.parameters(), lr=lr, weight_decay=1e-4)

    def check_coords(coord):
        # pass
        assert np.allclose((coord[0, :, 0] ** 2 + coord[1, :, 0] ** 2).detach().numpy(), 1), coord[:, 0, 0]

    # check_coords(coord_double_pend)
    # coord_double_pend = coord_double_pend + torch.randn_like(coord_double_pend) *noise  * coord_double_pend.std(
    #     dim=(0, 1))

    loss_best = 10000
    train_noise_std = .5 # TODO

    # training
    for epoch in range(epochs):
        # torch
        print(epoch)
        tuner.train(True)
        optimizer.zero_grad()

        # pendulums
        train_inits = torch.clamp(torch.randn(batch_size, 4).float().to(device), -1, 1) / 2.

        coord_double_pend = torch.from_numpy(return_coordinates_double_pendulum(
            double_pendulum, train_inits, ts, noise=0.))
        double_pendulum_approx_coordinates = DoublePendulumApproxDiffEqCoordinates(
            tuner=tuner,
            init=train_inits,
            external_force_1=external_force_1,
            external_force_2=external_force_2,
            method=method,
            ts=ts
        ).to(device)
        double_pendulum_approx_coordinates.reset(train_noise_std)
        coord_double_pend_approx = torch.stack(
            [double_pendulum_approx_coordinates(i) for i in range(len(ts))], 0).transpose(0, 1)
        # check_coords(coord_double_pend_approx)

        # loss and update
        loss = loss_fn(coord_double_pend, coord_double_pend_approx)
        loss.backward()
        optimizer.step()
        train_noise_std *= 0.995
        lr *= 0.999
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        experiment.log_metric('Train loss', loss.item(), step=epoch)

        # save pics every 50 epochs
        if epoch % logging_period == 0:

            # testing
            with torch.no_grad():
                # torch
                tuner.eval()
                tuner.train(False)

                # TODO: remove "_test"
                # pendulums
                test_inits = torch.clamp(torch.randn(batch_size, 4).float().to(device), -1, 1) / 2.
                double_pendulum_approx_coordinates_test = DoublePendulumApproxDiffEqCoordinates(
                    tuner=tuner,
                    init=test_inits,
                    external_force_1=external_force_1,
                    external_force_2=external_force_2,
                    method=method,
                    ts=ts
                ).to(device)
                double_pendulum_approx_coordinates_test.reset()
                coord_double_pend_approx = torch.stack(
                    [double_pendulum_approx_coordinates_test(i) for i in range(len(ts))], 0).transpose(0, 1)
                check_coords(coord_double_pend_approx)
                data_pendulum_approx = coord_double_pend_approx.numpy()
                data_pendulum = return_coordinates_double_pendulum(
                    double_pendulum, test_inits, ts, noise=0.)
                # loss_test_1 = np.sqrt(((data_pendulum_approx - data_pendulum) ** 2).mean())
                # print(data_pendulum_approx.shape, data_pendulum.shape)
                loss_test = loss_fn(torch.from_numpy(data_pendulum_approx), torch.from_numpy(data_pendulum))
                # print(loss_test_1, loss_test, "loss")

            # saving weights
            if loss_test.item() < loss_best:
                loss_best = loss_test.item()
                print(loss_best, end=' ')
                best_weights = copy.deepcopy(tuner.state_dict())
                torch.save(best_weights, open(PATH + 'tuner_{}.pcl'.format(experiment_key), 'wb+'))

            experiment.log_metric('Test loss', loss_test, step=epoch)

            with torch.no_grad():
                tuner.train(False)
                # double_pendulum_approx_coordinates_test.reset()
                # coord_double_pend_approx = torch.stack(
                #     [double_pendulum_approx_coordinates_test(i) for i in range(len(ts))], 0).transpose(0, 1)
                # data_pendulum_approx = coord_double_pend_approx.numpy()
                # data_pendulum = return_coordinates_double_pendulum(double_pendulum, test_inits, ts, noise=noise)
                fig = plot_pendulums(data_pendulum, data_pendulum_approx,
                                     double_pendulum_approx_coordinates_test._init_default_coords)
                experiment.log_figure(f"{epoch} Quality dynamic test", fig, step=epoch)
                plt.close()

                # double_pendulum_approx_coordinates.reset()
                # coord_double_pend_approx = torch.stack(
                #     [double_pendulum_approx_coordinates(i) for i in range(len(ts))], 0).transpose(0, 1)
                # data_pendulum_approx = coord_double_pend_approx.numpy()
                # data_pendulum = return_coordinates_double_pendulum(double_pendulum, train_inits, ts, noise=noise)
                # fig = plot_pendulums(data_pendulum, data_pendulum_approx)
                # experiment.log_figure(f"{epoch} Quality dynamic train", fig, step=epoch)
                # plt.close()


if __name__ == '__main__':
    main()
