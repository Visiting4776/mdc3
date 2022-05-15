from curses import window
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def cov(X: list[float],Y: list[float], bias: bool=True):
    # bias: if True, the denominator will be n, otherwise (n - 1).
    # see: https://math.stackexchange.com/q/2936143/
    return np.cov(X,Y, bias)[0][1]

def detrend(Y: list[float], deg: int=2) -> list[float]:
    # this function assumes all data points 
    # in the time series are equidistant.
    coefficients = np.polyfit(np.arange(len(Y)), Y, deg)
    trend = np.polyval(coefficients, np.arange(len(Y)))

    return Y - trend


def MDC3(X: pd.Series, Y: pd.Series, freq_min: int, freq_max: int, freq_step: int, sr: int, d: int):
    """ Calculate MDC3 correlation beteween two time series.
    Arguments:
        X: a time series
        Y: another time series
        freq_min: minimum frequency
        freq_max: maximum frequency
        freq_step: step size of the frequency rage
        sr: sampling rate. Window size = sr/frequency
        d: detrending degree
    Returns:
        MDC3 correlation between time series X and Y.
    """ 

    window_sizes = [sr/freq for freq in range(freq_min, freq_max+freq_step, freq_step)]
    print(window_sizes)
    pass


def DCCC(window_size: int, X: pd.Series, Y: pd.Series) -> float:
    if len(X) != len(Y):
        raise ValueError(f'time series must have same length ({len(X)} and {len(Y)} were given)')

    if len(X) % window_size > 0:
        raise ValueError(
            f'Time series of length {len(X)} not evenly divisible by window length {window_size}'
        )
    
    # Detrend both series on a per-window scale:
    detrended_x = [detrend(window) for window in np.split(X, len(X)/window_size)]
    detrended_y = [detrend(window) for window in np.split(Y, len(Y)/window_size)]

    return np.average(
        [cov(window_x, window_y) for window_x, window_y in zip(detrended_x, detrended_y)]
    ) / np.sqrt(
        np.average([np.var(w) for w in detrended_x]) *
        np.average([np.var(w) for w in detrended_y])
    )

if __name__ == '__main__':
    # eeg = pd.read_excel('sub-hc7_ses-hc.xls', header=None)
    eeg = pd.read_pickle('eegdata')
    window_size = 512

    # dccc = DCCC(window_size, X=eeg.iloc[:,0], Y=eeg.iloc[:,1])
    # print(dccc)
    mdc3 = MDC3(eeg.iloc[:,0], eeg.iloc[:,1], )