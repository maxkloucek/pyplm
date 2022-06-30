import numpy as np

# not sure whether to z-score or not?
def binarize(trajectory, TH=0):
    t_len, ROI_len = trajectory.shape
    mu = []  # global signal mean for each time point
    sigma = []  # std of global signal for each time point
    for t in range(0, t_len):
        mu.append(np.mean(trajectory[t, :]))
        sigma.append(np.std(trajectory[t, :]))
    mu = np.array(mu)
    sigma = np.array(sigma)

    z = np.zeros(t_len*ROI_len).reshape(t_len, ROI_len)
    S = np.zeros(t_len*ROI_len).reshape(t_len, ROI_len)
    for t in range(0, t_len):
        for ROI in range(0, ROI_len):
            z[t, ROI] = (trajectory[t, ROI] - mu[t]) / sigma[t]

    for t in range(0, t_len):
        for ROI in range(0, ROI_len):
            if z[t, ROI] >= TH:
                S[t, ROI] = 1
            else:  # i.e. < 0
                S[t, ROI] = -1
    return S