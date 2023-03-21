import numpy as np


def ising_interaction_matrix_1D_PBC(L, T, h, jval=1):
    """
    Generates a 1D Ising interaction matrix with periodic boundary conditions.

    Args:
        L (int): length of chain
        T (float): temperature
        h (float): external field (can be 1D array)
        jval (float, optional): coupling strength. Defaults to 1.

    Returns:
        model (np.array): interaction matrix, shape (L, L)

    Raises:
        ValueError: if h is not a scalar or 1D array of length L

    Example:
        >>> ising_interaction_matrix_1D_PBC(3, 1, 0)
        array([[ 0.,  1.,  0.],
                [ 1.,  0.,  1.],
                [ 0.,  1.,  0.]])
    """

    if np.isscalar(h):
        h = np.ones(L) * h
    elif len(h) != L:
        raise ValueError('h must be a scalar or 1D array of length L')

    model = np.zeros((L, L))
    for iN in range(0, L):
        i_left = iN - 1
        i_right = iN + 1
        if i_right == L:
            i_right = 0
        model[iN, i_left] = jval / T
        model[iN, i_right] = jval / T
    np.fill_diagonal(model, h / T)
    return model


def ising_interaction_matrix_2D_PBC(L, T, h, jval=1):

    N = L ** 2
    J = np.zeros((N, N))
    for i in range(0, N):
        if i % L == 0:
            J[i, i+L-1] = jval / T
        else:
            J[i, i-1] = jval / T

        if (i+1) % L == 0:
            J[i, i-L+1] = jval / T
        else:
            J[i, i+1] = jval / T

        if i < L:
            J[i, i + (N-L)] = jval / T
        else:
            J[i, i-L] = jval / T

        if i >= (N-L):
            J[i, i-(N-L)] = jval / T
        else:
            J[i, i+L] = jval / T
    np.fill_diagonal(J, h / T)
    return J


def generate_SK_model(N, T, h, jmean, jstd=1, seed=None):
    """
    Generates a Sherrington-Kirkpatric model.

    Args:
        N (int): number of spins
        T (float): temperature
        h (float or 1D vector): external fields
        jmean (float): mean of coupling strength
        jstd (float, optional=1): standard deviation of interaction strength.
        seed (int, optional): seed for random number generator

    Returns:
        model (np.array): interaction matrix, shape (N, N).
            Diagonal elements contain fields
            Off-diagonal elements contain coupling strengths

    Raises:
        ValueError: if h is not a scalar or 1D array of length N
    """
    if np.isscalar(h):
        h = np.ones(N) * h
    elif len(h) != N:
        raise ValueError('h must be a scalar or 1D array of length N')
    print(seed)
    rng = np.random.default_rng(seed=seed)

    jmean = jmean / N
    jstd = jstd / np.sqrt(N)
    # rand = np.random.normal(loc=jmean, scale=jstd, size=(N, N))
    rand = rng.normal(loc=jmean, scale=jstd, size=(N, N))
    J = np.tril(rand) + np.tril(rand, -1).T
    np.fill_diagonal(J, h)
    # so its all / T when I return it!!
    model = J / T
    print(model)
    return model
