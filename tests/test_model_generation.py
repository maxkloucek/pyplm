import pytest
import numpy as np

from pyplm import models


@pytest.mark.parametrize(
    'L, T, h, jval, expect',
    [(5, 1, 0, 1, [
        [0., 1., 0., 0., 1.],
        [1., 0., 1., 0., 0.],
        [0., 1., 0., 1., 0.],
        [0., 0., 1., 0., 1.],
        [1., 0., 0., 1., 0.]])
     ]
)
def test_1D_ising(L, T, h, jval, expect):
    print(expect)
    J = models.ising_interaction_matrix_1D_PBC(L, T, h, jval)
    assert np.allclose(J, expect)


def test_SK():
    N = 25
    T = 1
    h = 0
    jmean = 1
    J = models.generate_SK_model(N, T, h, jmean)
    assert J.shape == (N, N)
    assert np.allclose(J, J.T)
    assert np.allclose(np.diag(J), h / T)


# def test_all():
# need to add seed for the random number generator to test properly
# test without read writign!