import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()


def plot_pendulums(d1, d2, d3=None, component=0):
    fig, ax = plt.subplots(2, 2, figsize=(12, 12), dpi=200)
    titles = ["x1", "y1", "x2", "y2"]
    x = np.arange(d1.shape[1])
    for i in range(2):
        for j in range(2):
            ax[i][j].plot(x, d1[i * 2 + j, :, component], label='True')
            ax[i][j].plot(x, d2[i * 2 + j, :, component], label='Approx with DL')
            if d3 is not None:
                ax[i][j].plot(x, d3[i * 2 + j, :, component], label='Approx')
            ax[i][j].set_title(titles[i * 2 + j])
            ax[i][j].legend()
    return fig
