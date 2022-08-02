import numpy as np
from pyplm.utilities import tools
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def fit_power_laws(xy_arrays, ymin_value=1e-5):
    print(len(xy_arrays))
    # fig, ax = plt.subplots()
    xs_list = []
    yfit_list = []
    mcs = []
    r2s = []
    for xys in xy_arrays:
        xs = xys[0, :]
        ys = xys[1, :]
        # filter out bins with 0 content
        xs = xs[ys>ymin_value]
        ys = ys[ys>ymin_value]

        xs_log10 = np.log10(xs)
        ys_log10 = np.log10(ys)

        func_lin, mc = tools.curve_fit1D(tools.linear, xs_log10, ys_log10)
        # I should check the fit in log space really!

        yfit_log10 = func_lin(xs_log10, *mc)
        yfit = 10 ** yfit_log10
        r2s.append(r2_score(ys_log10, yfit_log10))
        # r2_adjusted = 1 - (1-r2) * ((len(yfit) - 1) / (len(yfit) - 2 - 1))
        # print(r2, r2_adjusted)
        
        # plt.plot(xs, ys)
        # ax.plot(xs, yfit, marker=',')
        mcs.append(mc)
        xs_list.append(xs)
        yfit_list.append(yfit)
    # ax.set(xscale='log', yscale='log')
    print(f'r2 min: {np.min(r2s):.3f}')
    # plt.show()
    return np.array(mcs), xs_list, yfit_list

# what do I wnat to do with the tail?
# hemisphere difference! i.e. lr rifferences
# in statepoints for certain individuals
# a summary, mu(h/J), sigma(h/J)
#            mu(tail), simga(tail)
# for