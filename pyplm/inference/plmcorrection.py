import numpy as np
from numba import njit
from scipy.optimize import minimize
from time import perf_counter
from ..simulate import multichain_sim
from .pseudolikelihoodmax import multimodel_inf


@njit
def correlation(trajectory):
    cij = np.cov(trajectory.T)
    N, _ = cij.shape
    C = np.sum(cij ** 2)
    C = C / N
    return C


def correction_C2(infmod_array, configs_array):
    if len(infmod_array) != len(configs_array):
        print('infmod and config array sample length missmatch in correction')

    method = 'Nelder-Mead'
    cormod_array = np.zeros_like(infmod_array)
    metadata = []
    for i, _ in enumerate(infmod_array):
        infmod = infmod_array[i]
        configs = configs_array[i]

        nSamples, nSpins = configs.shape
        C2_target = correlation(configs)
        print(i, nSamples, C2_target)
        t0 = perf_counter()
        # if i >= 84 and i < 231:
        res = minimize(
                cor_obj,
                x0=1.0,
                args=(infmod, C2_target, nSamples),
                method=method,
                options={
                    'disp': True, 'maxiter': 30,
                    # 'tol': 0.00001,
                    # 'xatol': 0.00001
                    }
            )
        x = res.x[0]
        func = res.fun
        # else:
        #     x = 1
        #     func = 100

        t1 = perf_counter()
        time = t1 - t0
        md = {'res': x, 'func': func, 'time': time}
        # else:
        #     x = 1
        #     md = {'res': 1, 'func': 100, 'time': 0}
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


def correction_jacknife(infmod_array, configs_array):
    nModels, B, N = configs_array.shape
    print(nModels, B, N)
    jacknife_models_array = np.zeros((nModels, B, N, N))
    metadata = []
    for i, _ in enumerate(infmod_array):
        model = infmod_array[i]
        configs = configs_array[i]
        jacknife_models, time_taken = jackknife(configs)
        metadata.append(time_taken)
        jacknife_models_array[i] = jacknife_models
    metadata = np.array(metadata, dtype=str)
    return jacknife_models_array, metadata


def jackknife(configurations):
    B, N = configurations.shape
    jacknife_indicies = np.arange(0, B)
    jacknife_models = np.zeros((B, N, N))
    t0 = perf_counter()
    for jacknife_index in jacknife_indicies:
        jacknife_configurations = [np.delete(configurations, jacknife_index, axis=0)]
        jacknife_configurations = np.array(jacknife_configurations)
        # print(jacknife_configurations.shape)
        infMod_array, times = multimodel_inf(jacknife_configurations, 6, 0)
        jacknife_models[jacknife_index, :, :] = infMod_array[0]
    t1 = perf_counter()
    metadata = f'JACKNIFE: Time taken = {t1-t0:3f}'
    return jacknife_models, metadata
