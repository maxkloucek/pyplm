import numpy as np
import matplotlib.pyplot as plt
import h5py

import pyplm.utilities as utils

def correlation(trajectory):
    cij = np.cov(trajectory.T)
    N, _ = cij.shape
    C = np.sum(cij ** 2)
    C = C / N
    return C


def triu_flat(matrix, k=1):
    triu_indices = np.triu_indices_from(matrix, k)
    upper_values = matrix[triu_indices]
    return upper_values


def temperature_estimate(model):
    N, _ = model.shape
    params = triu_flat(model)
    std = np.nanstd(params)
    temp = 1 / (std * N ** 0.5)
    return temp

plt.style.use("/Users/mk14423/Dropbox/custom.mplstyle")
file = 'testdata/test_whole.hdf5'
group = 'simtest2'
with h5py.File(file, 'r') as fin:
    conditions = list(fin.keys())
    group = fin[conditions[0]]
    datasets = list(group.keys())
    print(conditions)
    print(group)
    print(datasets)

    # print(trueMod_metaDF)
    # print(correction_metaDF)

    trueMod_array = group['inputModels'][()]
    infrMod_array = group['inferredModels'][()]
    corrMod_array = group['correctedModels'][()]
    trueMod_metaDF = utils.get_metadata_df(group, 'inputModels')
    correction_metaDF = utils.get_metadata_df(group, 'correctedModels')

    alphas = group['sweep-alphas'][()]
    sweep_trajectories = group['sweep-trajectories'][()]
    print(alphas.shape, sweep_trajectories.shape)

print(trueMod_metaDF)
print(correction_metaDF)
temps = trueMod_metaDF['T']
fig, ax = plt.subplots(3, len(trueMod_array))
if len(trueMod_array) == 1:
    ax = ax.reshape(3, 1)
print(ax.shape)
print(temps)

for iRun in range(0, len(trueMod_array)):
    label = temps[iRun]
    trueMod = trueMod_array[iRun]
    infrMod = infrMod_array[iRun]
    corrMod = corrMod_array[iRun]
    mods = [trueMod, infrMod, corrMod]
    estimated_temps = [temperature_estimate(mod) for mod in mods]
    print(estimated_temps)
    ax[0, iRun].imshow(trueMod)
    ax[1, iRun].imshow(infrMod)
    ax[2, iRun].imshow(corrMod)

    ax[0, iRun].set(title='T = {:.2f}'.format(label))
plt.show()

for alpha_trajectories in sweep_trajectories:
    C2s = []
    for trajectories in alpha_trajectories:
        C2_current = [correlation(trajectory) for trajectory in trajectories]
        C2_current = np.mean(C2_current)
        C2s.append(C2_current)
    plt.plot(alphas, C2s)
    plt.show()