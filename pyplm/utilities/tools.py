import numpy as np
from scipy.optimize import curve_fit


def linear(x, m, c):
    return (m*x) + c

# can I input an array here..?
def p1(x, b0, b1):
    return b0 + (b1 * x)

def p2(x, b0, b1, b2):
    return b0 + (b1 * x) + (b2 * (x ** 2))

def p3(x, b0, b1, b2, b3):
    return b0 + (b1 * x) + (b2 * (x ** 2)) + (b3 * (x ** 3))

def p_power(x, b0, b1, gamma):
    return b0 + (b1 * (x ** gamma))

def linear_origin(x, m):
    return (m*x)

def pure_power(x, a, gamma):
    return (a * (x ** gamma))

def sqrt_x(x, A, c):
    return (A * (x ** 0.5)) + c

# oh god well I really don't know.... it really doesn't seem to make much of a difference
# which function I fit. I should just accept it...?
def power(x, a, power, c):
    return (a * (x ** power)) + c

def arctan(x, A, B):
    return (A * np.arctan(B * x))


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

def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return a.flat[idx]

# -------------------------------------------------------------------------- #
