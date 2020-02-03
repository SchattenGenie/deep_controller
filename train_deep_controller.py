from comet_ml import Experiment
import torch
from torch import nn
from torchdiffeq import odeint_adjoint as odeint
from double_pendulum.double_pendulum import DoublePendulumDiffEq
from double_pendulum.double_pendulum_approximation import DoublePendulumApproxDiffEq, \
    DoublePendulumApproxDiffEqCoordinates
from coordinate_utils import return_coordinates_double_pendulum
from vizualization_utils import plot_pendulums
import matplotlib.pyplot as plt
from controller_tuner_nets import ControllerV1, ControllerExternalDerivativesV1, TunerCoordinatesV1
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
@click.option('--controller_type', type=str, default='None',
              help='None, internal or external_derivatives')
# @click.option('--is_tuner', type=bool, default=False)
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
        # is_tuner: bool
):
    controller_type = None if controller_type == 'None' else controller_type
    # TODO: experiments with deep controller & tuner
    # if not is_tuner and controller_type is None:
    #     print("Nothing to train.")
    #     return

    experiment = Experiment(project_name=project_name, workspace=work_space)
    experiment_key = experiment.get_key()
    PATH = './'
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    ts = torch.arange(start=0, end=duration, step=step).float().to(device)

    train_inits = torch.clamp(torch.randn(batch_size, 4).float().to(device), -1, 1) / 2.
    test_inits = torch.clamp(torch.randn(batch_size, 4).float().to(device), -1, 1) / 2.
    # TODO: mass, length, etc initializations in batch fashion
    # TODO: customization(i.e. several phase inits per mass init, several mass inits per phase init, force, etc

    if controller_type == "internal":
        controller = ControllerV1().to(device)
    elif controller_type == "external_derivatives":
        controller = ControllerExternalDerivativesV1().to(device)
    elif controller_type is None:
        controller = None
    else:
        raise ValueError(
            f"controller_type should be 'internal', 'external_derivatives' or 'None'."
            f" '{controller_type}' instead.")

    # tuner = TunerCoordinatesV1().to(device) if is_tuner else None

    double_pendulum = DoublePendulumDiffEq(
        external_force_1=external_force_1,
        external_force_2=external_force_2
    ).to(device)
    double_pendulum_approx = DoublePendulumApproxDiffEq(
        controller=controller,
        controller_type=controller_type,
        init=train_inits,
        external_force_1=external_force_1,
        external_force_2=external_force_2
    ).to(device)

    loss_fn = nn.MSELoss()
    if controller is not None:
        controller_optimizer = torch.optim.Adam(controller.parameters(), lr=lr, weight_decay=1e-4)
    # if tuner is not None:
    #     tuner_optimizer = torch.optim.Adam(tuner.parameters(), lr=lr, weight_decay=1e-4)

    # TODO: train both – controller and tuner?
    # optimizer = tuner_optimizer if is_tuner else controller_optimizer
    optimizer = controller_optimizer
    # best_weights = copy.deepcopy(tuner.state_dict()) if is_tuner else copy.deepcopy(controller.state_dict())
    best_weights = copy.deepcopy(controller.state_dict())

    coord_double_pend = odeint(double_pendulum, train_inits, ts, rtol=1e-3, atol=1e-3, method=method).detach().clone()
    coord_double_pend = coord_double_pend + torch.randn_like(coord_double_pend) * noise * coord_double_pend.std(
        dim=(0, 1))
    loss_best = 10000
    for epoch in range(epochs):
        optimizer.zero_grad()
        coord_double_pend_approx = odeint(double_pendulum_approx, train_inits, ts, rtol=1e-3, atol=1e-3, method=method)
        # if is_tuner:
        #     coord_double_pend_approx = tuner(coord_double_pend_approx)
        loss = loss_fn(coord_double_pend, coord_double_pend_approx)
        loss.backward()
        optimizer.step()

        # testing
        with torch.no_grad():
            double_pendulum_approx_test = DoublePendulumApproxDiffEq(
                controller=controller,
                controller_type=controller_type,
                init=test_inits,
                external_force_1=external_force_1,
                external_force_2=external_force_2
            ).to(device)
            data_pendulum_approx = return_coordinates_double_pendulum(double_pendulum_approx_test, test_inits, ts,
                                                                      noise=0.)
            data_pendulum = return_coordinates_double_pendulum(double_pendulum, test_inits, ts, noise=noise)
            # if is_tuner:
            #     data_pendulum_approx = tuner(torch.from_numpy(data_pendulum_approx)).numpy()
            loss_test = np.sqrt(((data_pendulum_approx - data_pendulum) ** 2).mean())

        # saving weights
        if loss_test.item() < loss_best:
            loss_best = loss_test.item()
            print(loss_best, end=' ')
            # if is_tuner:
            #     best_weights = copy.deepcopy(tuner.state_dict())
            #     torch.save(best_weights, open(PATH + 'tuner_{}.pcl'.format(experiment_key), 'wb+'))
            # else:
            #     best_weights = copy.deepcopy(controller.state_dict())
            #     torch.save(best_weights, open(PATH + 'controller_{}.pcl'.format(experiment_key), 'wb+'))

            best_weights = copy.deepcopy(controller.state_dict())
            torch.save(best_weights, open(PATH + 'controller_{}.pcl'.format(experiment_key), 'wb+'))

        experiment.log_metric('Train loss', loss.item(), step=epoch)
        experiment.log_metric('Test loss', loss_test, step=epoch)

        # save pics every 50 epochs
        if epoch % logging_period == 0:
            with torch.no_grad():
                data_pendulum_approx = return_coordinates_double_pendulum(double_pendulum_approx_test, test_inits, ts,
                                                                          noise=0.)
                data_pendulum = return_coordinates_double_pendulum(double_pendulum, test_inits, ts, noise=noise)
                # if is_tuner:
                #     data_pendulum_approx = tuner(data_pendulum_approx)
                fig = plot_pendulums(data_pendulum, data_pendulum_approx)
                experiment.log_figure("Quality dynamic test", fig, step=epoch)
                plt.close()

                data_pendulum_approx = return_coordinates_double_pendulum(double_pendulum_approx, train_inits, ts,
                                                                          noise=0.)
                data_pendulum = return_coordinates_double_pendulum(double_pendulum, train_inits, ts, noise=noise)
                # if is_tuner:
                #     data_pendulum_approx = tuner(data_pendulum_approx)
                fig = plot_pendulums(data_pendulum, data_pendulum_approx)
                experiment.log_figure("Quality dynamic train", fig, step=epoch)
                plt.close()


if __name__ == '__main__':
    main()
