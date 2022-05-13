import numpy as np
# from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt
import random

def cov(X,Y):
    # mean_X = sum(X)/len(X)
    # mean_Y = sum(Y)/len(Y)
    # return np.mean([
    #     (x - mean_X)*(y - mean_Y) 
    #     for x, y in zip(X, Y)
    # ])
    return np.cov(X,Y, ddof=0)[0][1]


def detrend(X: list[float], Y: list[float], deg: int=2) -> list[float]:
    coefficients = np.polyfit(X, Y, deg)
    predicted = np.polyval(coefficients, X)
    detrended = Y - predicted

    fig, axes = plt.subplots(nrows=2, sharex=True)
    axes[0].plot(X, Y, 'k')
    axes[0].plot(X, predicted, 'b')
    axes[0].set(title = f'Original series and {deg}. degree fit')

    axes[1].plot(X, detrended, 'k')
    axes[1].set(title = f'Detrended series')

    plt.show()


def mdc3(X, Y, freq_min, freq_max, freq_step, sr, d):
    pass


if __name__ == '__main__':
    x = np.arange(50)
    y = [y+random.uniform(-3,3) for y in np.linspace(22, 28, 50)]

    detrend(x, y)