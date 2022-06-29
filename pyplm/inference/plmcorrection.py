import numpy as np
from numba import njit
from scipy.optimize import minimize
from time import perf_counter
from ..simulate import multichain_sim

import matplotlib.pyplot as plt

# this function needs to go elsewhere, e.g. into something
# called observables?
# observables.trajectory
# observables.thermodynamicAverages
@njit
def correlation(trajectory):
    cij = np.cov(trajectory.T)
    N, _ = cij.shape
    C = np.sum(cij ** 2)
    C = C / N
    return C


def correction(infmod_array, configs_array):
    if len(infmod_array) != len(configs_array):
        print('infmod and config array sample length missmatch in correction')

    method='Nelder-Mead'
    cormod_array = np.zeros_like(infmod_array)
    metadata = []
    for i, _ in enumerate(infmod_array):
        infmod = infmod_array[i]
        configs = configs_array[i]
    
        nSamples, nSpins = configs.shape
        C2_target = correlation(configs)
        print(C2_target)
        # something dodgy about correction for 2D ising...
        t0 = perf_counter()
        res = minimize(
                cor_obj,
                x0=1.0,
                args=(infmod, C2_target, nSamples),
                method=method,
                options={
                    'disp': True, 'maxiter': 50,
                    # 'tol': 0.00001,
                    # 'xatol': 0.00001
                    }
            )
        t1 = perf_counter()
        time = t1 - t0
        x = res.x[0]
        md = {'res': x, 'func': res.fun, 'time': time}
        print(md)
        metadata.append(md)
        cormod_array[i] = infmod / x
    metadata = np.array(metadata, dtype=str)
    return cormod_array, metadata


def cor_obj(alpha, init_model, C2_target, B_target):
    # B, N = traget_traj.shape
    # C_target = correlation(traget_traj)
    model = init_model / (alpha)
    trajectories = multichain_sim(
        model, 6, 0, 1e4, B_target, 6)

    C2_current = [correlation(trajectory) for trajectory in trajectories]
    C2_current = np.mean(C2_current)
    print('C2 current: {}, C2 target: {}'.format(C2_current, C2_target))
    objective = (C2_target - C2_current) ** 2
    return objective

# then also do a T sweep function based / smilar to this!