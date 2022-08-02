import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from inference.core import utils
from joblib import Parallel
from matplotlib.colors import LogNorm
from scipy.sparse import csr_matrix

import inference.scripts.optimizeT_funcs as opT


def correlation_sim(model, B):
    nReps = 6
    vSet = 0
    saveFreq = 1e1
    cyclesEQ = 1e3 * saveFreq
    cyclesPROD = B * saveFreq
    # traj = fullRun_serial(model, cyclesEQ, cyclesPROD, saveFreq)
    # corr = correlation(traj)

    with Parallel(n_jobs=nReps, verbose=vSet) as parallel:
        trajs = opT.fullRun(
            parallel, model, nReps, cyclesEQ, cyclesPROD, saveFreq)
        # corr_reps = np.array([correlation(traj) for traj in trajs])
    # corr = np.mean(corr_reps)
    return trajs


# only works for odd sized models I think!
def collapseRadially_periodic(matrix):
    matrix = np.copy(matrix)
    N, _ = matrix.shape
    for i in range(1, N):
        matrix[i, :] = np.roll(matrix[i, :], -i)
        # Cij[iS, :] = np.roll(Cij[iS, :], -iS)

    # fig, ax = plt.subplots()
    # ax.matshow(matrix)
    # plt.show()

    Mr = np.mean(matrix, axis=0)
    Mr_positive = Mr[1: int(nSpins / 2) + 1]
    Mr_negative = np.flip(Mr[int(nSpins / 2) + 1:])
    Mr = (Mr_positive + Mr_negative) / 2

    r = np.arange(0, Mr.size) + 1
    return Mr, r

plt.style.use("/Users/mk14423/Dropbox/custom.mplstyle")
Jij = utils.ising_interaction_matrix_1D_PBC(N=11, T=1, h=0, jval=1)

trajs = correlation_sim(Jij, 1e4)

nReps, nSamples, nSpins = trajs.shape
full_traj = np.zeros((nReps * nSamples, nSpins))

for iR, traj in enumerate(trajs):
    full_traj[iR * nSamples: (iR + 1) * nSamples, :] = traj
