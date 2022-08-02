import numpy as np
import matplotlib.pyplot as plt
from pyplm.pipelines import model_pipeline, data_pipeline
from pyplm.analyse import tail

plt.style.use("/Users/mk14423/Dropbox/custom.mplstyle")

file = '/Users/mk14423/Desktop/PaperData/HCP_data_Ryota/HCP_rsfmri.hdf5'
group = 'individuals'
# group = 'grouped'
dataset = 'inferredModels'

# pipe = data_pipeline(file, group, dataset)
# print(len(pipe.mods))

# print(pipe.mods['left'].shape)
# hs_indi, Js_indi = pipe.get_parameters('individuals')
# # print(pipe.__dict__.items())
# exit()
fig, ax = plt.subplots()
modpipe = model_pipeline(file, group, dataset)
# print(modpipe.datasets.shape)
# plt.matshow(modpipe.datasets[0])
# plt.show()
modpipe.split_hemispheres(which='left')
# print(modpipe.datasets.shape)
J_meanstds = modpipe.select_parameters(parameter_name='J')
# fig, ax = plt.subplots()
# ax.plot(J_meanstds[0, :], J_meanstds[1, :], ls='none')
# plt.show()
modpipe.transform_to_distributions(
    filter_level=0, bins=100, density=True)

modpipe.tail_distributions(tail='positive')
mcs, xs, yfit = tail.fit_power_laws(modpipe.datasets)
nMods = 3  # len(modpipe.datasets)
for iMod in range(0, nMods):
    xy = modpipe.datasets[iMod]
    x = xy[0, :]
    y = xy[1, :]
    ax.plot(x, y, label=f'm: {mcs[iMod, 0]:.2f} c {mcs[iMod, 1]:.2f}')
    ax.plot(xs[iMod], yfit[iMod], marker=',', c='k', ls='--')

group = 'grouped'
modpipe = model_pipeline(file, group, dataset)
modpipe.split_hemispheres(which='left')
J_meanstds = modpipe.select_parameters(parameter_name='J')
modpipe.transform_to_distributions(
    filter_level=0, bins=100, density=True)

modpipe.tail_distributions(tail='positive')
mcs, xs, yfit = tail.fit_power_laws(modpipe.datasets)
for iMod in range(0, len(modpipe.datasets)):
    xy = modpipe.datasets[iMod]
    x = xy[0, :]
    y = xy[1, :]
    ax.plot(x, y, label=f'm: {mcs[iMod, 0]:.2f} c {mcs[iMod, 1]:.2f}')
    ax.plot(xs[iMod], yfit[iMod], marker=',', c='k')
plt.xlabel(r'$J_{ij}$')
plt.ylabel(r'$P(J_{ij})$')
plt.legend()
plt.show()

modpipe.tail_distributions(tail='positive')
mcs, xs, yfit = tail.fit_power_laws(modpipe.datasets)
fig, ax = plt.subplots()
for iMod in range(0, len(modpipe.datasets)):
    xy_tail = modpipe.datasets[iMod]
    print(xy_tail.shape)
    x = xy_tail[0, :]
    y = xy_tail[1, :]
    ax.plot(x, y)
plt.show()

print(mcs.shape)
fig, ax = plt.subplots(2, 1)
ax = ax.ravel()
mean = np.mean(mcs[:, 0])
std = np.std(mcs[:, 0])
lbl = f'm: {mean:.2f} pm {std:.2f}'
ax[0].hist(mcs[:, 0], bins=20, label=lbl)
ax[0].legend()

mean = np.mean(mcs[:, 1])
std = np.std(mcs[:, 1])
lbl = f'c: {mean:.2f} pm {std:.2f}'
ax[1].hist(mcs[:, 1], bins=20, label=lbl)
ax[1].legend()
plt.show()
# distros = modpipe.parameter_distributions(Js, 0, bins=101, density=True)
# for distro in distros:
#     plt.plot(distro[0], distro[1])
#     plt.show()
# hs_meanstd = modpipe.parameter_moments(hs)
# Js_meanstd = modpipe.parameter_moments(Js)
# # plt.plot(hs_meanstd[0, :],hs_meanstd[1, :], ls='none')
# plt.plot(Js_meanstd[0, :],Js_meanstd[1, :], ls='none')
# plt.show()
# # print(modpipe.Js.shape, modpipe.hs.shape)
# # print(modpipe.models.shape)