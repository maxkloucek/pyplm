# do the temperature sweep
import numpy as np
import matplotlib.pyplot as plt
from pyplm.simulate import multichain_sim


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
            )
        )
    print(sweep_trajectories.shape)
    for iMod, mod in enumerate(mod_array):        
        for iAlpha, alpha in enumerate(alphas):
            trajectories = multichain_sim(
                    mod / alpha, 6, 1, 1e4, nSamples, nChains)
            sweep_trajectories[
                iMod, iAlpha, :] = trajectories

    # print(sweep_trajectories[-1, -1, -1, 10:20, 5])
    return sweep_trajectories


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