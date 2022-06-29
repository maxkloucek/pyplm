import os
import h5py
import numpy as np
from joblib.externals.loky import get_reusable_executor

import pyplm.utilities.hdf5io as io
import pyplm.inference as inference
from pyplm import simulate as sim
from pyplm import models

import matplotlib.pyplot as plt

# pipeline will be a class.
# it's faster to run functions outside of class definition is what
# I've noticed, i.e. have them in their own helper space!
# should all write to a hdf5 file, and a group in that file
# that's how I've decided it works best!
# always have nDims = 3 so that I can save a sweep as well!
class pipeline:
    def __init__(self, file_name, group_name):
        self.fname = file_name
        self.gname = group_name
        os.makedirs(os.path.dirname(self.fname), exist_ok=True)

    def generate_model(
            self, mod_choices, mod_args_list,
            mod_name='inputModels'):
        # "inputModels"
        if len(mod_choices) != len(mod_choices):
            print('length of choices and args are missmatched!')
            exit()
        mod_array = []
        mod_metadata = []
        for mod_choice, mod_args in zip(mod_choices, mod_args_list):
            if mod_choice == '1D_ISING_PBC':
                model = models.ising_interaction_matrix_1D_PBC(**mod_args)
                mod_array.append(model)
            elif mod_choice == '2D_ISING_PBC':
                model = models.ising_interaction_matrix_2D_PBC(**mod_args)
                mod_array.append(model)
            elif mod_choice == 'SK':
                model = models.SK_interaction_matrix(**mod_args)
                mod_array.append(model)
            else:
                print(
                    'Note, non-preset model choice made, supply your own?')
                model = None
                mod_array.append(model)
            model_md = {'model': mod_choice}
            model_md.update(mod_args)
            mod_metadata.append(model_md)

        # have to all be same shape for this to work!
        mod_array = np.array(mod_array)
        mod_metadata_array = np.array(mod_metadata, dtype=str)
        print('--Saving Models--')
        print(mod_metadata_array)
        print('-----------------')
        io.write_models_to_hdf5(
            self.fname, self.gname, mod_name, mod_array, mod_metadata_array)
        # return mod_array, mod_metadata_array

    def simulate_data(self, sim_args_list, n_jobs=-1, verbose=5):
        print('-Simulating Data-')
        mod_array, mod_md = io.get_models(
            self.fname, self.gname, 'inputModels')
        if len(mod_md) != len(sim_args_list):
            print('did not supply enough sim args to sim all models')
            exit()

        trajectories, sim_md = sim.multimodel_sim(
            mod_array, sim_args_list, n_jobs, verbose)

        io.write_configurations_to_hdf5(
            self.fname,
            self.gname,
            trajectories,
            sim_md
        )
        print('-----------------')

    # expects an extra frist dimension, incase multiple runs # labels!
    def write_data(self, datasets, labels):
        print(self.fname)
        io.write_configurations_to_hdf5(
            self.fname,
            self.gname,
            datasets,
            labels)

    def infer(
            self,
            nParallel_jobs=6, Parallel_verbosity=0,
            mod_name='inferredModels'):
        get_reusable_executor().shutdown(wait=True)
        print('---Running PLM---')

        configs_array = io.get_configurations(self.fname, self.gname)
        mod_array, mod_md = inference.multimodel_inf(
            configs_array, nParallel_jobs, Parallel_verbosity)
        print(mod_array.shape, mod_md)
        io.write_models_to_hdf5(
            self.fname,
            self.gname,
            mod_name,
            mod_array,
            mod_md
        )
        print('-----------------')

    def correct_plm(
            self, mod_name='correctedModels'):
        print('Running correction')
        infmod_array, _ = io.get_models(
            self.fname, self.gname, 'inferredModels')
        configs_array = io.get_configurations(self.fname, self.gname)
        cormod_array, cormod_md = inference.correction(infmod_array, configs_array)
        # have a look at the md of the other things
        # and save something similar, i.e. json dictionary string!
        # sweet it works!
        print(cormod_md)
        io.write_models_to_hdf5(
            self.fname,
            self.gname,
            mod_name,
            cormod_array,
            cormod_md
        )
        print('-----------------')

    # currently cannot vary alpha across models!
    def ficticiousT_sweep(
            self, alphas,
            nSamples, nChains,
            mod_name='inferredModels'):
        mod_array, _ = io.get_models(
            self.fname, self.gname, mod_name)
        # might have to change to corrected Models!
        # for full analysis
        sweep_trajectories = sim.TempSweep(mod_array, alphas, nSamples, nChains)
        print(sweep_trajectories.shape)
        io.write_sweep_to_hdf5(
            self.fname, self.gname,
            alphas, sweep_trajectories)