import numpy as np
from pyplm import models


def test_SK():
    N = 4
    T = 1
    h = 0
    jmean = 1
    seed = 0
    J = models.generate_SK_model(N, T, h, jmean, seed=seed)
    assert J.shape == (N, N)
    assert np.allclose(J, J.T)
    assert np.allclose(np.diag(J), h / T)


test_SK()