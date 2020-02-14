import numpy as np


def generate_length(low=0.3, high=1.):
    length_1 = np.random.uniform(low, high)
    length_2 = np.random.uniform(low, high)
    return length_1, length_2
