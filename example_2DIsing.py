import numpy as np
from pyplm.pipelines import data_pipeline

file = './2DIsing_example.hdf5'
group = '2d_ising_N25'

# pipeline is designed to perform inference on multiple models
# this behaviour can be mimicked by setting e.g. Ts = np.linspace(0.1, 3, 10)

Ts = [2.5]  # generate a single model at temperature T = 2.5
# possible mod_choices '1D_ISING_PBC', '2D_ISING_PBC', 'SK'
mod_choices = ['2D_ISING_PBC' for _ in Ts]
mod_args = [{'L': 5, 'T': T, 'h': 0, 'jval': 1} for T in Ts]
sim_args = [{'B_eq': 2e3, 'B_sample': 1e3, 'nChains': 6} for _ in Ts]

# initialise pipeline
plm_pipeline = data_pipeline(file, group)

# generate model, and simulate data
plm_pipeline.generate_model(mod_choices, mod_args)
plm_pipeline.simulate_data(sim_args, n_jobs=6)

# alternatively, add data manualy to pipeline.
# Data-object has to be numpy array, shape (nModels, B, N)
# nModels: no. models, B: no. samples, N: no. spins
# plm_pipeline.write_data(data, labels)

# inference options
plm_pipeline.infer(Parallel_verbosity=5)  # performs PLM inference
plm_pipeline.correct_firth(mod_name='firthModels')  # Firth's correction
plm_pipeline.correct_C2(mod_name='c2Models')  # self-consistency correction

plm_pipeline.ficticiousT_sweep(
    np.linspace(0.1, 3, 100), 5e3, 6)
