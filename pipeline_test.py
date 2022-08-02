import os
import numpy as np
import matplotlib.pyplot as plt
from pyplm.pipelines import data_pipeline
import json
import h5py
import glob


def get_sample_trajectories(dataset):
    files = glob.glob(os.path.join(dataset, '*'))
    files = np.array(files)
    config_datasets = []
    for iF in (range(0, len(files))):
        time_series = np.loadtxt(files[iF], delimiter=',')
        spin_trajectory = np.copy(time_series)
        spin_trajectory[spin_trajectory <= 0] = -1
        spin_trajectory[spin_trajectory > 0] = 1
        config_datasets.append(spin_trajectory)
        # print(files[iF], spin_trajectory.shape)
    return files, np.array(config_datasets)

def get_random_trajectories(B, N):
    data = np.random.randint(low=0, high=2, size=(B,N))
    data[data == 0] = -1
    data = np.array([data])
    labels = np.array(['dataset1'])
    return labels, data


plt.style.use("/Users/mk14423/Dropbox/custom.mplstyle")
# labels, data = get_sample_trajectories(
#     '/Users/mk14423/Desktop/PaperData/HCP_data_Ryota/allRest')
labels, data = get_random_trajectories(B=500, N=10)
print(labels, data.shape)
file = '/Users/mk14423/Desktop/Data/0_Thesis/example_data_small.hdf5'
group = '1D_ising_N101'
# with h5py.File(file, 'a') as fin:
#     print(fin.keys())
#     del fin[group]
# exit()
# mod_choices = [
#         # '1D_ISING_PBC',
#         # '2D_ISING_PBC',
#         'SK',
# ]
# mod_args = [
#     # {'L': 64, 'T': 3, 'h': 0, 'jval': 1},
#     # {'L': 20, 'T': 25, 'h': 0, 'jval': 1,},
#     {'N': 128, 'T': 1.5, 'h': 0, 'jmean': 0.3, 'jstd': 1,},
# ]
# sim_args = [
#     {'B_eq': 1e4, 'B_sample': 1e3, 'nChains': 1}
#     ]

Ts = [1]
mod_choices = ['1D_ISING_PBC' for _ in Ts]
mod_args = [{'L': 101, 'T': T, 'h': 0, 'jval': 1} for T in Ts]
sim_args = [{'B_eq': 1e4, 'B_sample': 1e4, 'nChains': 6} for _ in Ts]


# plm_pipeline.write_data(data, labels)
plm_pipeline = data_pipeline(file, group)
plm_pipeline.generate_model(mod_choices, mod_args)
plm_pipeline.simulate_data(sim_args, n_jobs=6)
plm_pipeline.infer(Parallel_verbosity=5)
# plm_pipeline.correct_plm()
# plm_pipeline.ficticiousT_sweep(
#     np.linspace(0.1, 4, 200), 1e4, 6)

# let's do a whole pipline test!
# and then do a sweep

# file = 'testdata/test1.hdf5'
# group = 'simtest'
# Ts = [0.5, 1, 2.25, 5]
# mod_choices = ['2D_ISING_PBC' for _ in Ts]
# mod_args = [{'L': 7, 'T': T, 'h': 0, 'jval': 1} for T in Ts]
# sim_args = [{'B_eq': 1e3, 'B_sample': 5e3, 'nChains': 6} for _ in Ts]
# # print(mod_args)
# print(len(mod_args), len(sim_args))
# plm_pipeline = pipeline(file, group)
# plm_pipeline.generate_model(mod_choices, mod_args)
# plm_pipeline.simulate_data(sim_args, n_jobs=6)
# # plm_pipeline.write_data(data, labels)
# plm_pipeline.infer(Parallel_verbosity=5)


# file = 'testdata/test1.hdf5'
# group = 'htest_lowcoupling'
# hs = np.linspace(-4, +4, 100)
# mod_choices = ['2D_ISING_PBC' for _ in hs]
# mod_args = [{'L': 7, 'T': 1, 'h': h, 'jval': 0.1} for h in hs]
# sim_args = [{'B_eq': 1e3, 'B_sample': 5e3, 'nChains': 6} for _ in hs]
# # print(mod_args)
# print(len(mod_args), len(sim_args))
# plm_pipeline = pipeline(file, group)
# plm_pipeline.generate_model(mod_choices, mod_args)
# plm_pipeline.simulate_data(sim_args, n_jobs=6)
# # plm_pipeline.write_data(data, labels)
# plm_pipeline.infer(Parallel_verbosity=5)