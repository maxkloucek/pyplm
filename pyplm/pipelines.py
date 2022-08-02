import os
import h5py
import numpy as np
from joblib.externals.loky import get_reusable_executor
from scipy.interpolate import UnivariateSpline

import pyplm.utilities.hdf5io as io
import pyplm.inference as inference
from pyplm import simulate as sim
from pyplm import models
from pyplm.utilities import tools

import matplotlib.pyplot as plt

# pipeline will be a class.
# it's faster to run functions outside of class definition is what
# I've noticed, i.e. have them in their own helper space!
# should all write to a hdf5 file, and a group in that file
# that's how I've decided it works best!
# always have nDims = 3 so that I can save a sweep as well!
class data_pipeline:
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



from pyplm.analyse import models as ma

# pipeline that takes in models and transforms them along
# the way? that probably makes it make the most sense?
# so call it dataset? also I use the word models too often!
# i.e. cluttered namespace!
class model_pipeline:
    def __init__(self, file_name, group_name, model_name):
        self.fname = file_name
        self.gname = group_name
        self.dname = model_name

        with h5py.File(file_name, 'r') as fin:
            group = fin[group_name]
            metadata = group[model_name + '_metadata'].asstr()[()]
            models = group[model_name][()]
        # models = models[0:3]
        self.datasets = models
        self.md = metadata
        n1, n2, n3 = models.shape
        if n2 != n2:
            print('array does not contain square matricies')
            exit()

    def split_hemispheres(self, which='both'):
        mods_ll, mods_rr = ma.get_split_hemispheres(
            self.datasets)
        if which == 'left':
            self.datasets = mods_ll
        if which == 'right':
            self.datasets = mods_rr
        if which == 'both':
            print('unchanged')

    def select_parameters(self, parameter_name):
        hs, Js = ma.split_parameters(self.datasets)
        if parameter_name == 'h':
            self.datasets = hs
        if parameter_name == 'J':
            self.datasets = Js

        means = np.nanmean(self.datasets, axis=1)
        stds = np.nanstd(self.datasets, axis=1)
        return np.vstack((means, stds))

    def transform_to_distributions(
        self, filter_level, **histargs):
        distributions = []
        for params in self.datasets:
            # print(params.shape)
            x, y = ma.centered_histogram(
                params, filter_level, **histargs)
            # I am smoothing here to make it more nice!
            # remove filter level thing now?
            # cause not really necessary anymore?
            y = np.convolve(y, np.ones(5)/5, mode='same')
            # print(y.shape)
            distributions.append([x, y])
        self.datasets = np.array(distributions)

    def tail_distributions(self, tail='positive'):
        r1s = []
        r2s = []
        xy_tail = []
        # xy_negative = []
        # xy_positive = []
        for dataset in self.datasets:
            x = dataset[0, :]
            y = dataset[1, :]
            
            spline = UnivariateSpline(x, y-(np.max(y)/2), s=0)
            # i maybe want to find a more robust way of
            # defining the tail?
            r1, r2 = spline.roots() # find the roots
            r1s.append(r1)
            r2s.append(r2)
            # print(r1, r2)
            if tail == 'negative':
                x_tail = x[x <= r1]
                y_tail = y[x <= r1]

            if tail == 'positive':
                x_tail = x[x > r2]
                y_tail = y[x > r2]
                # x_core
                # y_core
                # work out the mean
                # from a histogram?
                # this seems unreliable?
            xy_tail.append(np.array([x_tail, y_tail]))
        self.datasets = xy_tail
        return np.array(r1s), np.array(r2s)

# namespaces are messing up!
class NOdata_pipeline:
    def __init__(self, file_name, group_name, model_name):
        self.fname = file_name
        self.gname = group_name
        self.dname = model_name

        with h5py.File(file_name, 'r') as fin:
            group = fin[group_name]
            metadata = group[model_name + '_metadata'].asstr()[()]
            mods = group[model_name][()]
        # models = models[0:3]
        # these should really be dictionaries
        # right?
        mods_left, mods_right = ma.get_split_hemispheres(mods)
        self.mods = {
            'full': mods,
            'left': mods_left,
            'right': mods_right}
        self.mods_means = {
            'full': np.array([np.mean(mods, axis=0)]),
            'left': np.array([np.mean(mods_left, axis=0)]),
            'right': np.array([np.mean(mods_right, axis=0)])}

        self.md = metadata

    def get_parameters(self, means_or_individuals):
        if means_or_individuals == 'means':
            mods = self.mods_means
            for mod_array in mods:
                print('----')
                print(mod_array.shape)
                hs, Js = ma.split_parameters(mod_array)
                print(hs.shape, Js.shape)
        if means_or_individuals == 'individuals':
            mods = self.mods
            for mod_array in mods:
                print('----')
                print(mod_array.shape)
                hs, Js = ma.split_parameters(mod_array)
                print(hs.shape, Js.shape)

    def transform_to_distributions(
        self, filter_level, **histargs):
        distributions = []
        for params in self.datasets:
            # print(params.shape)
            x, y = ma.centered_histogram(
                params, filter_level, **histargs)
            # I am smoothing here to make it more nice!
            # remove filter level thing now?
            # cause not really necessary anymore?
            y = np.convolve(y, np.ones(5)/5, mode='same')
            # print(y.shape)
            distributions.append([x, y])
        self.datasets = np.array(distributions)

    def tail_distributions(self, tail='positive'):
        r1s = []
        r2s = []
        xy_tail = []
        # xy_negative = []
        # xy_positive = []
        for dataset in self.datasets:
            x = dataset[0, :]
            y = dataset[1, :]
            
            spline = UnivariateSpline(x, y-(np.max(y)/8), s=0)
            # i maybe want to find a more robust way of
            # defining the tail?
            r1, r2 = spline.roots() # find the roots
            r1s.append(r1)
            r2s.append(r2)
            # print(r1, r2)
            if tail == 'negative':
                x_tail = x[x <= r1]
                y_tail = y[x <= r1]

            if tail == 'positive':
                x_tail = x[x > r2]
                y_tail = y[x > r2]
                # x_core
                # y_core
                # work out the mean
                # from a histogram?
                # this seems unreliable?
            xy_tail.append(np.array([x_tail, y_tail]))
        self.datasets = xy_tail
        return np.array(r1s), np.array(r2s)