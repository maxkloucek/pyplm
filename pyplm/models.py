import numpy as np


def ising_interaction_matrix_1D_PBC(L, T, h, jval=1):
    J = np.zeros((L, L))
    for iN in range(0, L):
        i_left = iN - 1
        i_right = iN + 1
        if i_right == L:
            i_right = 0
        J[iN, i_left] = jval / T
        J[iN, i_right] = jval / T
    np.fill_diagonal(J, h / T)
    return J


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


def SK_interaction_matrix(N, T, h, jmean, jstd=1):
    jmean = jmean / N
    jstd = jstd / np.sqrt(N)
    rand = np.random.normal(loc=jmean, scale=jstd, size=(N, N))
    J = np.tril(rand) + np.tril(rand, -1).T
    np.fill_diagonal(J, h)
    # so its all / T when I return it!!
    J = J / T
    return J
