import numpy as np
import matplotlib.pyplot as plt
import h5py
from pyplm.analyse import statphyis
from pyplm.utilities import tools
# let's append the pervious result I got for inferred model form ryota data!
plt.style.use("/Users/mk14423/Dropbox/custom.mplstyle")

dataset_dictionary_0 = {
    'file': '/Users/mk14423/Desktop/PaperData/HCP_data_Ryota/HCP_rsfmri.hdf5',
    'group': 'individuals',
    'dsname': 'inferredModels',
    'pname':'J'
}
dataset_dictionary_1 = {
    'file': '/Users/mk14423/Desktop/PaperData/HCP_data_Ryota/HCP_rsfmri.hdf5',
    'group': 'grouped',
    'dsname': 'inferredModels',
    'pname':'J'
}
ds_dic = [dataset_dictionary_0, dataset_dictionary_1]
# fig, ax = plt.subplots()
# statphyis.Jstatepoints(ax, ds_dic, ls='none')
# plt.show()

# fig, ax = plt.subplots()
# statphyis.Jtails(ax, ds_dic)
# ax.set(xlabel=r'$J_{ij}$', ylabel=r'$P(J_{ij})$')
# plt.legend()
# plt.tight_layout(pad=0)
# plt.show()

# fig, ax = plt.subplots(2, 2)
# mods = statphyis.Jij_avrg_vs_groupLR(ax, ds_dic)
# plt.tight_layout(pad=0)
# plt.show()
# print(len(mods), mods[0].shape)
# print(np.mean(mods[0] / mods[2]))
# print(np.mean(mods[1] / mods[3]))
# do the sweep for each of these and see what I feel about it??
# I've left a complete mess here...

# plot what the mean model would look like from the
# individuals as well!!!
# what about the taiiils?

# this is not very clean aahahhahaeguheurgurgu
# what am I doooiiing


fig, ax = plt.subplots()
individuals = statphyis.JStatepoints(**dataset_dictionary_0)
# individuals.statepoints_overview(ax, ls='none', c='r')
# meanModel = individuals.statepoints_overview_avrgMod(ax, c='c')
individuals.statepoints_overview_splitHem(ax, ls='none', label='individuals')
individuals.statepoints_overview_splitHemMean(ax, ls='none', label='mean')

dataset_dictionary_0['group'] = 'grouped'
grouped = statphyis.JStatepoints(**dataset_dictionary_0)
# groupModel = grouped.statepoints_overview(ax, ls='none', c='b')
grouped.statepoints_overview_splitHem(ax, ls='none', label='grouped')
plt.legend(title='Left / Right Hemispheres')
plt.show()
# meanModel = meanModel[0]
# groupModel = groupModel[0]
# fig, ax = plt.subplots(1, 3)
# ax = ax.ravel()
# ax[0].matshow(meanModel)
# ax[1].matshow(groupModel)
# ax[2].matshow(groupModel-meanModel)
# # plt.show()
# fig, ax = plt.subplots()

# ax.hist(
#     tools.triu_flat(meanModel, k=0) -tools.triu_flat(groupModel, k=0),
#     bins=100)
# plt.show()
# print(meanModel.shape, groupModel.shape)
# divide it over something?
# statphyis.statepoints_overview(ax, dataset_dictionary, ls='none', c='r')
# statphyis.statepoints_overview_splitH(ax, dataset_dictionary, ls='none', c='r')

# # statphyis.statepoints_overview(ax, dataset_dictionary, ls='none', c='b')
# statphyis.statepoints_overview_splitH(ax, dataset_dictionary, ls='none', c='r')
