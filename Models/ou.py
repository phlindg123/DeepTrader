import numpy as np


def OU(theta, sigma):
    state = 0
    while True:
        yield state
        state += -theta*state + sigma*np.random.randn()
