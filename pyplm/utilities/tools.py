import numpy as np
from scipy.optimize import curve_fit


def linear(x, m, c):
    return (m*x) + c


def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x-x0)**2/(2*sigma**2))


def curve_fit1D(func, xdata, ydata):
    """Fitting a arbitrary 1D curve"""
    popt, pcov = curve_fit(func, xdata, ydata)
    # xfit = np.linspace(xdata.min(), xdata.max(), 50)
    # yfit = func(xfit, *popt)
    return func, popt

def fit_func1D(func, xdata, ydata):
    """Fitting a arbitrary 1D curve"""
    popt, pcov = curve_fit(func, xdata, ydata)
    # xfit = np.linspace(xdata.min(), xdata.max(), 50)
    yfit = func(xdata, *popt)
    return yfit, popt

def triu_flat(matrix, k=1):
    triu_indices = np.triu_indices_from(matrix, k)
    upper_values = matrix[triu_indices]
    return upper_values