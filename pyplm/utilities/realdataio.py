import glob
import os
import numpy as np
from pyplm.preprocess import binarize

def load_configurations_from_csvs(data_directory):
    files = glob.glob(os.path.join(data_directory, '*'))
    files = files[0:5]
    # print(files)
    # nDatasets = len(files)
    # print(nDatasets)
    configurations_array = []
    for iF in (range(0, len(files))):
        # print(files[iF])
        time_series = np.loadtxt(files[iF], delimiter=',')
        spin_trajectory = binarize(time_series)
        print(files[iF], spin_trajectory.shape)
        configurations_array.append(spin_trajectory)
    configurations_array = np.array(configurations_array)
    return configurations_array


