import numpy as np
import h5py

from sklearn.linear_model import LogisticRegression
from joblib import Parallel
from joblib import delayed
from time import perf_counter



def multimodel_inf(configurations_array, n_jobs, verbose):
    # print(infMod_ds)
    nMods, nSamples, nSpins = configurations_array.shape
    print(nMods, nSamples, nSpins)

    infMod_array = np.zeros((nMods, nSpins, nSpins))
    times = np.zeros(nMods, dtype=str)
    print(infMod_array.shape)

    with Parallel(
        n_jobs=n_jobs,
        verbose=verbose,
        # timeout=72000
        ) as parallel:
        for i, configs in enumerate(configurations_array):
            t0 = perf_counter()
            infMod = plm(configs, parallel)
            infMod_array[i, :, :] = infMod
            t1 = perf_counter()
            print(i, configs.shape, infMod.shape)
            print('Time take: {:.3f}s'.format(t1-t0))
            times[i] = str(t1-t0)
    return infMod_array, times

def plm(trajectory, parallel):
    # print(trajectory.shape)
    _, nFeatures = trajectory.shape
    model_inf = parallel(
        delayed(logRegLoop_inner)(trajectory, row_index)
        for row_index in range(0, nFeatures)
    )
    model_inf = np.array(model_inf)
    model_inf = (model_inf + model_inf.T) * 0.5
    # print(model_inf.shape)
    return model_inf


def logRegLoop_inner(traj, row_index):
    nSamples, nSpins = traj.shape
    X = np.delete(traj, row_index, 1)
    y = traj[:, row_index]  # target
    log_reg = LogisticRegression(
        penalty='none',
        # C=10,
        # random_state=0,
        solver='lbfgs',
        max_iter=200)

    log_reg.fit(X, y)
    weights = log_reg.coef_[0] / 2
    # factor of 2 from equations! wieghts size = N-1
    bias = log_reg.intercept_[0] / 2

    left_weights = weights[0:row_index]
    right_weights = weights[row_index:]

    row_parameters = np.zeros(nSpins)
    row_parameters[0:row_index] = left_weights
    row_parameters[row_index+1:] = right_weights
    row_parameters[row_index] = bias
    return row_parameters