import os
import h5py


def write_models_to_hdf5(
        file_name, group_name,
        mod_name,
        mod_array, mod_md_array):

    if os.path.exists(file_name) is True:
        fopen_kwrds = {
            'mode': 'a',
        }
        # open_mode = 'a'
    else:
        fopen_kwrds = {
            'mode': 'w',
            'track_order': True,
            'fs_strategy': 'fsm',
            'fs_persist': True
        }
    print(fopen_kwrds)
    with h5py.File(file_name, **fopen_kwrds) as fout:
        group = fout.require_group(group_name)

        model_ds = group.require_dataset(
            mod_name,
            dtype=mod_array.dtype,
            shape=mod_array.shape,
            compression="gzip")
        model_ds[()] = mod_array

        string_dtype = h5py.string_dtype(encoding='utf-8', length=None)
        mod_labels_ds = group.require_dataset(
            mod_name + "_metadata",
            shape=mod_md_array.shape,
            dtype=string_dtype,
            )
        mod_labels_ds[()] = mod_md_array


def get_models(file_name, group_name, mod_name):
    with h5py.File(file_name, 'r') as fout:
        mod_array = fout[group_name + '/' + mod_name][()]
        mod_md = fout[
            group_name + '/' + mod_name + '_metadata'].asstr()[()]
    return mod_array, mod_md


# maybe I should always have this check? nah
def write_configurations_to_hdf5(
        file_name, group_name, datasets, labels):
    if os.path.exists(file_name) is True:
        fopen_kwrds = {
            'mode': 'a',
        }
        open_mode = 'a'
    else:
        fopen_kwrds = {
            'mode': 'w',
            'track_order': True,
            'fs_strategy': 'fsm',
            'fs_persist': True
        }
    print(fopen_kwrds)
    with h5py.File(
                file_name,
                **fopen_kwrds
                ) as fout:

        group = fout.require_group(group_name)

        string_dtype = h5py.string_dtype(encoding='utf-8', length=None)
        labels_ds = group.require_dataset(
            "configurations_metadata",
            shape=labels.shape,
            dtype=string_dtype,
            )
        labels_ds[()] = labels

        config_ds = group.require_dataset(
            "configurations",
            shape=datasets.shape,
            dtype=datasets.dtype,
            compression="gzip")
        config_ds[()] = datasets


def write_sweep_to_hdf5(
        file_name, group_name,
        temps, sweep_trajectories):
    with h5py.File(file_name, 'a') as fout:
        group = fout.require_group(group_name)
        print('Saving sweep, dtype is', sweep_trajectories.dtype)
        alphas_ds = group.require_dataset(
            "sweep-temps",
            shape=temps.shape,
            dtype=temps.dtype,
            compression="gzip")
        alphas_ds[()] = temps

        sweep_ds = group.require_dataset(
            "sweep-trajectories",
            shape=sweep_trajectories.shape,
            dtype=sweep_trajectories.dtype,
            compression="gzip", compression_opts=9)
        sweep_ds[()] = sweep_trajectories


# setting up the sweep properly!

def get_configurations(file_name, group_name):
    with h5py.File(file_name, 'r') as fin:
        configs_ds = fin[group_name + '/configurations']
        # this needs a bit of work, at the moment my sims
        # just ato discard the eq, so this will never
        # execute!
        if "eq_cycles" in dict(configs_ds.attrs.items()):
            discard = int(
                configs_ds.attrs["eq_cycles"] /
                configs_ds.attrs["cycle_dumpfreq"])
            config_array = configs_ds[discard:, :]
        else:
            config_array = configs_ds[()]
        nDatasets, nSamples, nSpins = config_array.shape
        # print(nDatasets, nSamples, nSpins)
        return config_array


# infMod_ds = fin[group_name].require_dataset(
#     'inferredModels',
#     shape=(nDatasets, nSpins, nSpins),
#     dtype=float,
#     compression='gzip')
