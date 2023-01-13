import numpy as np
# import matplotlib.pyplot as plt
from pyplm.pipelines import data_pipeline

# import glob

# plt.style.use("/Users/mk14423/Dropbox/custom.mplstyle")
# labels, data = get_sample_trajectories(
#     '/Users/mk14423/Desktop/PaperData/HCP_data_Ryota/allRest')

file = './2DIsing_example.hdf5'
group = '2d_ising_N16'

Ts = [3]
# possible mod_choices '1D_ISING_PBC', '2D_ISING_PBC', 'SK'
mod_choices = ['2D_ISING_PBC' for _ in Ts]
mod_args = [{'L': 4, 'T': T, 'h': 0, 'jval': 1} for T in Ts]
sim_args = [{'B_eq': 2e3, 'B_sample': 1e3, 'nChains': 6} for _ in Ts]

plm_pipeline = data_pipeline(file, group)
plm_pipeline.generate_model(mod_choices, mod_args)
plm_pipeline.simulate_data(sim_args, n_jobs=6)
plm_pipeline.infer(Parallel_verbosity=5)
plm_pipeline.correct_firth(mod_name='firthModels')
plm_pipeline.correct_C2(mod_name='c2Models')
# plm_pipeline.correct_plm()
plm_pipeline.ficticiousT_sweep(
    np.linspace(0.1, 4, 1), 1e3, 6)


# plm_pipeline.write_data(data, labels)
# alterantively we can write data.

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