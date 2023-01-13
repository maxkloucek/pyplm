import numpy as np
from . import basemc as mc
from joblib import Parallel, delayed


def multimodel_sim(mod_array, sim_args_list, n_jobs, verbose):
        trajectories = []
        for model, sim_args in zip(mod_array, sim_args_list):
            print(sim_args)
            trajs = multichain_sim(model, n_jobs, verbose, **sim_args)
            nChains, nSamples, nSpins = trajs.shape
            combined_trajectory = np.zeros((nChains * nSamples, nSpins))
            for iC in range(0, nChains):
                combined_trajectory[
                    (iC) * nSamples : (iC + 1) * nSamples, :] = trajs[iC, :, :]
            trajectories.append(combined_trajectory)

        trajectories = np.array(trajectories)
        sim_args_array = np.array(sim_args_list, dtype=str)
        return trajectories, sim_args_array


# auto discards B_eq!
def multichain_sim(model, n_jobs, vset, B_eq, B_sample, nChains):
    # nReps = 6
    # vSet = 0
    # it's got the B_eq in here!! make sure you're sensible when changing this!
    # saving every N steps now!
    saveFreq = 1e1 # 1e1
    cyclesEQ = B_eq * saveFreq
    cyclesPROD = B_sample * saveFreq
    # traj = fullRun_serial(model, cyclesEQ, cyclesPROD, saveFreq)
    # corr = correlation(traj)

    with Parallel(n_jobs=n_jobs, verbose=vset) as parallel:
        trajs = fullRun(
            parallel, model, nChains, cyclesEQ, cyclesPROD, saveFreq)
    return trajs

# automtically dicards the eq trajectories!
def fullRun(parallel_instance, model, nReps, cyclesEQ, cyclesPROD, saveFreq):
    N, _ = model.shape
    cycles = int(cyclesEQ + cyclesPROD)
    nSamplesPROD = int(cyclesPROD / saveFreq)

    initFunc = mc.initialise_ising_config
    iConfigs = [initFunc(N, 0)for _ in range(0, nReps)]
    simArgs = genParaSimArgs(
                model, cycles, saveFreq, iConfigs)
    sim_trajectories = sim_parallel(parallel_instance, sim_core, simArgs)
    PROD_trajectories = sim_trajectories[:, -nSamplesPROD:]
    return PROD_trajectories


def genParaSimArgs(model, mcCycles, mcSaveFreq, configsInit):
    args = [
        [model, mcCycles, mcSaveFreq, configInit] for configInit in configsInit]
    return np.array(args, dtype=object)


def sim_parallel(joblib_parallel_object, func, func_arg_list):
    fout = joblib_parallel_object(
            delayed(sim_core)(*tuple(args)) for args in (func_arg_list)
        )
    return np.array(fout)


def sim_core(model, cycles, saving_freq, init_config):
    configs, _ = mc.simulate(model, init_config, cycles, saving_freq)
    # final_config = configs[-1]
    return configs




