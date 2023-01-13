import numpy as np


def greedy_subsample(array, subsample_size):
    max_full_subsamples = int(array.size / subsample_size)
    subsamples = np.zeros((max_full_subsamples, subsample_size))

    remaining_pool = np.copy(array)
    for i in range(0, max_full_subsamples):
        subsample, remaining_pool = subsample_indices(
                remaining_pool, subsample_size)
        subsamples[i, :] = subsample
    return subsamples


def subsample_indices(array, n_subsamples):
    indices = np.arange(0, array.size)
    subsampled_indicies = np.random.choice(
            indices, n_subsamples, replace=False)

    mask = np.ones(indices.shape, dtype=bool)
    mask[subsampled_indicies] = False
    remaining_indicies = indices[mask]
    return array[subsampled_indicies], array[remaining_indicies]


def get_configurations(subsample_indicies, trajectory):
    n_ss, n_ss_samples = subsample_indicies.shape
    n_samples, N = trajectory.shape

    subsampled_trajectories = np.zeros((n_ss, n_ss_samples, N))

    for i_ss in range(0, n_ss):
        indicies = subsample_indicies[i_ss].astype(int)
        subsampled_trajectories[i_ss, :, :] = trajectory[indicies, :]
    return subsampled_trajectories
