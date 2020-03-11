import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()


def plot_pendulums(d_true=None, d_dc=None, d_approx=None, d_tuner=None, component=0):
    # TODO: args as dict {d_name: d_data, ...}
    fig, ax = plt.subplots(2, 2, figsize=(12, 12), dpi=200)
    titles = ["x1", "y1", "x2", "y2"]
    x = np.arange(d_true.shape[1])
    print(d_true.shape)
    for i in range(2):
        for j in range(2):
            ax[i][j].plot(x, d_true[i * 2 + j, :, component], label='True')
            if d_approx is not None:
                assert d_approx.shape == d_true.shape
                ax[i][j].plot(x, d_approx[i * 2 + j, :, component], label='Approx')
            if d_dc is not None:
                assert d_dc.shape == d_true.shape
                ax[i][j].plot(x, d_dc[i * 2 + j, :, component], label='Approx with DC')
            if d_tuner is not None:
                assert d_tuner.shape == d_true.shape
                ax[i][j].plot(x, d_tuner[i * 2 + j, :, component], label='Approx with Tuner')
            ax[i][j].set_title(titles[i * 2 + j])
            ax[i][j].legend()
    return fig
