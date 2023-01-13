import os
import h5py
import numpy as np
import pandas as pd
from joblib.externals.loky import get_reusable_executor
from time import perf_counter

import pyplm.utilities.hdf5io as io
import pyplm.inference as inference
from pyplm import simulate as sim
from pyplm import models
from pyplm.utilities import tools
from pyplm import utilities
from pyplm.simulate import multichain_sim


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
        t0 = perf_counter()
        configs_array = io.get_configurations(self.fname, self.gname)
        mod_array, mod_md = inference.multimodel_inf(
            configs_array, nParallel_jobs, Parallel_verbosity)

        io.write_models_to_hdf5(
            self.fname,
            self.gname,
            mod_name,
            mod_array,
            mod_md
        )
        t1 = perf_counter()
        print('-----------------')
        print(f'PLM time taken: = {t1-t0:.3f}s')

    def correct_firth(
            self,
            nParallel_jobs=6, Parallel_verbosity=0,
            mod_name='firthModels'):
        get_reusable_executor().shutdown(wait=True)
        print('---Running Firth Correction ---')
        t0 = perf_counter()
        configs_array = io.get_configurations(self.fname, self.gname)
        mod_array, mod_md = inference.multimodel_inf_firth(
            configs_array, nParallel_jobs, Parallel_verbosity)

        io.write_models_to_hdf5(
            self.fname,
            self.gname,
            mod_name,
            mod_array,
            mod_md
        )
        t1 = perf_counter()
        print('-----------------')
        print(f'Firth Correction time taken: = {t1-t0:.3f}s')

    def correct_C2(
            self, mod_name='c2Models'):
        print('Running C2 Correction')
        t0 = perf_counter()
        infmod_array, _ = io.get_models(
            self.fname, self.gname, 'inferredModels')
        configs_array = io.get_configurations(self.fname, self.gname)
        cormod_array, cormod_md = inference.correction_C2(
                infmod_array, configs_array)
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
        t1 = perf_counter()
        print('-----------------')
        print(f'C2 time taken: = {t1-t0:.3f}s')

    def correct_jackknife(
            self, mod_name='correctedModels_jacknife'):
        print('Running Jacknife Correction')
        t0 = perf_counter()
        infmod_array, _ = io.get_models(
            self.fname, self.gname, 'inferredModels')
        configs_array = io.get_configurations(self.fname, self.gname)

        jacknife_models_array, jn_metadata = inference.correction_jacknife(
                infmod_array, configs_array)

        io.write_models_to_hdf5(
            self.fname,
            self.gname,
            mod_name,
            jacknife_models_array,
            jn_metadata
        )
        t1 = perf_counter()
        print('-----------------')
        print(f'Jackknife time taken: = {t1-t0:.3f}s')

    # currently cannot vary alpha across models!
    def ficticiousT_sweep(
            self, alphas,
            nSamples, nChains,
            mod_name='inferredModels'):
        mod_array, _ = io.get_models(
            self.fname, self.gname, mod_name)
        # might have to change to corrected Models!
        # for full analysis
        sweep_trajectories = sim.TempSweep(
                mod_array, alphas, nSamples, nChains)
        io.write_sweep_to_hdf5(
            self.fname, self.gname,
            alphas, sweep_trajectories)

    def threshold_sweep(
            self, nSamples, nChains,
            nThresholds=50,
            symmetric=True,
            mod_name='inferredModels', sweep_gname='sweepTH_symmetric'):

        nSamples_eq = int(1e3)
        nSamples = int(nSamples)

        mod_array, _ = io.get_models(
            self.fname, self.gname, mod_name)
        print(mod_array.shape)

        nMods, N, N = mod_array.shape
        if nMods != 1:
            print('WHATCH OUT, IVE ONLY WRITTEN THIS CODE TO WORK WITH 1MOD!')
            exit()

        base_model = mod_array[0, :, :]
        min = np.abs(base_model).min()
        min = min - (0.01 * min)
        logmin = np.log10(min)
        max = np.abs(base_model).max()
        logmax = np.log10(max)
        thresholds = np.logspace(logmin, logmax, nThresholds)

        print(thresholds.shape)
        sweep_traj_shape = (nThresholds, nChains, nSamples, N)
        print(sweep_traj_shape)

        with h5py.File(self.fname, 'a') as fout:
            print(fout[self.gname].keys())
            g = fout[self.gname].require_group(sweep_gname)
            param_ds = g.require_dataset(
                "parameters",
                shape=thresholds.shape,
                dtype=thresholds.dtype,
                compression="gzip", compression_opts=9)
            param_ds[()] = thresholds
            g.require_dataset(
                "trajectories",
                shape=sweep_traj_shape,
                dtype=float,
                compression="gzip", compression_opts=9)

        for i_th, th in enumerate(thresholds):
            t0 = perf_counter()
            mod = np.copy(base_model)
            N, _ = mod.shape
            if symmetric is True:
                mod[np.abs(mod) <= th] = 0

            trajectories = multichain_sim(
                        mod, 6, 1, nSamples_eq, nSamples, nChains)
            with h5py.File(self.fname, 'a') as fout:
                # print(fout[self.gname][sweep_gname]['trajectories'].shape)
                fout[self.gname][sweep_gname]['trajectories'][i_th] = (
                    trajectories)
            t1 = perf_counter()
            print(
                    f'i_th: {i_th} simulating time taken: {t1-t0:.3f}s')

    def subsample(
            self, no_ss_points):
        print('Subsampling analysis...')

        configs_array = io.get_configurations(self.fname, self.gname)
        nDatasets,  nSamples, nSpins = configs_array.shape
        print(nDatasets, nSamples, nSpins)
        ss_slice_lengths = int(nSamples / no_ss_points)
        print(ss_slice_lengths)
        # nDatasets = 2
        # # no_ss_points = 1
        possible_sample_indicies = np.arange(0, nSamples, dtype=int)
        iDs = []
        Ns = []
        n_subsamples = []
        means_J = []
        stds_J = []
        for iD in range(0, nDatasets):
            print(f'--- Dataset: {iD} ---')
            for i_ss in range(1, no_ss_points + 1):

                subsample_size = ss_slice_lengths * i_ss
                print(i_ss, subsample_size, no_ss_points + 1)
                subsamples = utilities.greedy_subsample(
                    possible_sample_indicies,
                    subsample_size
                )
                ss_trajectories = utilities.subsampling.get_configurations(
                    subsamples, configs_array[iD]
                )
                # print(subsample_size, ss_trajectories.shape)
                mod_array, mod_md = inference.multimodel_inf(
                    ss_trajectories, 6, 0)
                # print(mod_array.shape, mod_md)

                # ravel through my things and append each time...
                n_repeats, _, _ = mod_array.shape
                for iR in range(0, n_repeats):
                    iDs.append(iD)
                    Ns.append(nSpins)
                    n_subsamples.append(subsample_size)
                    Js = tools.triu_flat(mod_array[iR, :, :], k=1)
                    means_J.append(np.mean(Js))
                    stds_J.append(np.std(Js))

        obs_dict = {
            'iD': iDs,
            'N': Ns,
            'B': n_subsamples,
            'mean_J': means_J,
            'std_J': stds_J
        }
        dataframe = pd.DataFrame(
            data=obs_dict
        )
        print(dataframe)
        key = self.gname + '/subsampling'
        print(key)
        dataframe.to_hdf(self.fname, key=key, mode='a', complevel=5)
        print('-----------------')
