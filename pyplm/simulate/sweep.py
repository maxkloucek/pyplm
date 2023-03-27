# do the temperature sweep
import numpy as np
from pyplm.simulate import multichain_sim
import pyplm.simulate.basemc as mc


def TempSweep(mod_array, alphas, nSamples, nChains):
    nModels, nSpins, _ = mod_array.shape
    nSweepStatepoints = alphas.size
    # print(nModels, nSpins, nSweepStatepoints, nSamples)
    sweep_trajectories = np.zeros(
        (
            nModels,
            nSweepStatepoints,
            nChains,
            int(nSamples),
            nSpins
            ),
        dtype=np.int8
        )
    print(sweep_trajectories.shape, sweep_trajectories.dtype)
    for iMod, mod in enumerate(mod_array):  
        for iAlpha, alpha in enumerate(alphas):
            trajectories = multichain_sim(
                    mod / alpha, 6, 1, 1e4, nSamples, nChains)
            sweep_trajectories[
                iMod, iAlpha, :] = trajectories

    # print(sweep_trajectories[-1, -1, -1, 10:20, 5])
    return sweep_trajectories


# from pyplm.utilities.tools import triu_flat
# import matplotlib.pyplot as plt

# def ThresholdSweep(base_model, nThresholds, nSamples, nChains, symmetric=True):
#     # NOT USED XX
#     # pre_th_mod = np.copy(model)
#     # mod = model
#     nSamples = 1e3
#     min = np.abs(base_model).min()
#     min = min - (0.01 * min)
#     logmin = np.log10(min)

#     max = np.abs(base_model).max()
#     logmax = np.log10(max)

#     thresholds = np.logspace(logmin, logmax, nThresholds)
#     thresholds = thresholds[-5:-4]
#     # thresholds = np.logspace(logmin, logmax, 5)
    
#     nConnections = []
#     for th in thresholds:
#         mod = np.copy(base_model)
#         N, _ = mod.shape
#         if symmetric == True:
#             mod[np.abs(mod) <= th] = 0

#         trajectories = multichain_sim(
#                     mod, 6, 1, 0.5 * 1e3, nSamples, nChains)
#         # print(trajectories.shape)
#         # plt.matshow(trajectories[0, :, :])
#         # plt.show()
#         # I thne want to save this for each threshold time!

#         # params = triu_flat(mod)
#         # plt.hist(params, bins='auto')
#         # pdf, x = np.histogram(params, bins='auto', density=True)
#         # x = x[:-1]
#         # ax.plot(x, pdf)
#         # ax.axvline(th, marker=',', c='k')
#         # plt.show()
#         # mod[np.abs(mod) > th] = 1
#         nConnections.append(np.sum(mod == 0))
#     # ax.set(xlim=[thresholds[0], None], xscale='log', yscale='log')
#     # plt.show()
#     return thresholds, nConnections
#     # I can make all the trajectories and then save them that makes the most sense I think


# def correction(infmod_array, configs_array):
#     if len(infmod_array) != len(configs_array):
#         print('infmod and config array sample length missmatch in correction')
#     # alpha_correction_array = np.zeros()
#     # = 
#     for infmod, configs in zip(infmod_array, configs_array):
#         nSamples, nSpins = configs.shape
#         print(infmod.shape, configs.shape)
#         C2_target = correlation(configs)
#         print(C2_target)
#         alphas = np.linspace(0.8, 1.5, 20)
#         objs = []
#         for alpha in alphas:
#             obj = cor_obj(alpha, infmod, C2_target, nSamples)
#             objs.append(obj)
#         print(objs)
#         plt.plot(alphas, objs)
#         plt.show()
#     # return alpha_correction_array


# def cor_obj(alpha, init_model, C2_target, B_target):
#     # B, N = traget_traj.shape
#     # C_target = correlation(traget_traj)
#     model = init_model / (alpha)
#     trajectories = multichain_sim(
#         model, 6, 1, 1e4, B_target, 6)
#     print(trajectories.shape)
#     C2_current = [correlation(trajectory) for trajectory in trajectories]
#     C2_current = np.mean(C2_current)
#     print(C2_current)
#     objective = (C2_target - C2_current) ** 2
#     return objective