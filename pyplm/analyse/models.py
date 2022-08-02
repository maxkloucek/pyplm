import numpy as np

def triu_flat(matrix, k=1):
    triu_indices = np.triu_indices_from(matrix, k)
    upper_values = matrix[triu_indices]
    return upper_values


def get_split_hemispheres(models, N_hemispheres=180):
    # models, _ = get_models(file, group, name)
    models_ll = models[:, 0:N_hemispheres, 0:N_hemispheres]
    models_rr = models[:, N_hemispheres:, N_hemispheres:]
    return models_ll, models_rr


def split_parameters(models):
    hs = np.array([np.diag(model) for model in models])
    Js = np.array([triu_flat(model) for model in models])
    return hs, Js


def centered_histogram(data, filter_level, **histargs):
    n, bin_edges = np.histogram(data, **histargs)
    bin_centers = bin_edges[:-1] + 0.5 * (bin_edges[1] - bin_edges[0])
    # print(filter_level)
    if filter_level is not None:
        for level in range(0, filter_level):
            mins = np.min(n)
            # print(mins, n[n == mins].size)
            bin_centers = bin_centers[n > mins]
            n = n[n > mins]

            # print(mins, bin_centers.shape, n.shape)
    return bin_centers, n
