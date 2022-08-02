import numpy as np
import matplotlib.pyplot as plt

from pyplm.pipelines import model_pipeline
from pyplm.utilities import tools

class JStatepoints:
    def __init__(self, file, group, dsname, pname):
        self.file = file
        self.group = group
        self.dsname = dsname
        self.pname = pname
    
    def statepoints_overview(self, ax, **pltargs):
        modpipe = model_pipeline(
            self.file, self.group, self.dsname)
        models = np.copy(modpipe.datasets )
        _, _, N = modpipe.datasets.shape
        J_meanstdsL = modpipe.select_parameters(self.pname)
        left_means = J_meanstdsL[0, :] * N
        left_stds = J_meanstdsL[1, :] * (N ** 0.5)
    
        ax.plot(left_means / left_stds, 1 / left_stds, **pltargs)
        ax.set(
            xlabel=r'$\mu _{J} / \sigma_{J}$',
            ylabel=r'$1 / \sigma_{J}$')
        return models

    def statepoints_overview_avrgMod(self, ax, **pltargs):
        modpipe = model_pipeline(
            self.file, self.group, self.dsname)
        modpipe.datasets = np.array(
            [np.mean(modpipe.datasets, axis=0)])
        models = np.copy(modpipe.datasets)
        _, _, N = modpipe.datasets.shape
        J_meanstdsL = modpipe.select_parameters(self.pname)
        left_means = J_meanstdsL[0, :] * N
        left_stds = J_meanstdsL[1, :] * (N ** 0.5)
    
        ax.plot(left_means / left_stds, 1 / left_stds, **pltargs)
        ax.set(
            xlabel=r'$\mu _{J} / \sigma_{J}$',
            ylabel=r'$1 / \sigma_{J}$')

        return models

    def statepoints_overview_splitHemMean(self, ax, **pltargs):
        modpipeL = model_pipeline(
            self.file, self.group, self.dsname)
        modpipeL.datasets = np.array(
            [np.mean(modpipeL.datasets, axis=0)])
        modpipeL.split_hemispheres(which='left')
        _, _, N = modpipeL.datasets.shape
        J_meanstdsL = modpipeL.select_parameters(self.pname)
        # modpipeL.transform_to_distributions(0, bins=100, density=True)
        # modpipeL.tail_distributions(tail='positive')
        # mcsL = tail.fit_power_laws(modpipeL.datasets)

        modpipeR = model_pipeline(
            self.file, self.group, self.dsname)
        modpipeR.datasets = np.array(
            [np.mean(modpipeR.datasets, axis=0)])
        modpipeR.split_hemispheres(which='right')
        J_meanstdsR = modpipeR.select_parameters(self.pname)
        # modpipeR.transform_to_distributions(0, bins=100, density=True)
        # modpipeR.tail_distributions(tail='positive')
        # mcsR = tail.fit_power_laws(modpipeR.datasets)

        left_means = J_meanstdsL[0, :] * N
        left_stds = J_meanstdsL[1, :] * (N ** 0.5)
        right_means = J_meanstdsR[0, :] * N
        right_stds = J_meanstdsR[1, :] * (N ** 0.5)
        # fig, ax = plt.subplots(2, 1, figsize=(6,5))
        # ax = ax.ravel()
        # fig, ax = plt.subplots(figsize=(6,5))
        # ax = [ax]
        # connect two of the same thing with a line?
        # then this defines some error distance
        # in two2 space?
        ax.plot(left_means / left_stds, 1 / left_stds, **pltargs)
        ax.plot(right_means / right_stds, 1 / right_stds, **pltargs)
        for iROI in range(0, len(left_stds)):
            ax.plot(
                [left_means / left_stds, right_means / right_stds],
                [1 / left_stds, 1 / right_stds],
                c='k', marker=','
                )
        ax.set(
            xlabel=r'$\mu _{J} / \sigma_{J}$',
            ylabel=r'$1 / \sigma_{J}$')

    def statepoints_overview_splitHem(self, ax, **pltargs):
        modpipeL = model_pipeline(
            self.file, self.group, self.dsname)
        modpipeL.split_hemispheres(which='left')
        _, _, N = modpipeL.datasets.shape
        J_meanstdsL = modpipeL.select_parameters(self.pname)
        # modpipeL.transform_to_distributions(0, bins=100, density=True)
        # modpipeL.tail_distributions(tail='positive')
        # mcsL = tail.fit_power_laws(modpipeL.datasets)

        modpipeR = model_pipeline(
            self.file, self.group, self.dsname)
        modpipeR.split_hemispheres(which='right')
        J_meanstdsR = modpipeR.select_parameters(self.pname)
        # modpipeR.transform_to_distributions(0, bins=100, density=True)
        # modpipeR.tail_distributions(tail='positive')
        # mcsR = tail.fit_power_laws(modpipeR.datasets)

        left_means = J_meanstdsL[0, :] * N
        left_stds = J_meanstdsL[1, :] * (N ** 0.5)
        right_means = J_meanstdsR[0, :] * N
        right_stds = J_meanstdsR[1, :] * (N ** 0.5)
        # fig, ax = plt.subplots(2, 1, figsize=(6,5))
        # ax = ax.ravel()
        # fig, ax = plt.subplots(figsize=(6,5))
        # ax = [ax]
        # connect two of the same thing with a line?
        # then this defines some error distance
        # in two2 space?
        ax.plot(left_means / left_stds, 1 / left_stds, **pltargs)
        ax.plot(right_means / right_stds, 1 / right_stds, **pltargs)
        for iROI in range(0, len(left_stds)):
            ax.plot(
                [left_means / left_stds, right_means / right_stds],
                [1 / left_stds, 1 / right_stds],
                c='k', marker=','
                )
        ax.set(
            xlabel=r'$\mu _{J} / \sigma_{J}$',
            ylabel=r'$1 / \sigma_{J}$')
        # and a whole brain differnece tomorrow!

        # ax[1].plot(left_stds, mcsL[:, 0], ls='none')
        # ax[1].plot(right_stds, mcsR[:, 0], ls='none')
        # ax[1].plot(left_means, mcsL[:, 0], ls='none')
        # ax[1].plot(right_means, mcsR[:, 0], ls='none')
        # for iROI in range(0, len(left_stds)):
        #     ax[1].plot(
        #         [left_stds, right_stds],
        #         [mcsL[:, 0], mcsR[:, 0]],
        #         c='k', marker=',', alpha=0.5
        #         )
        # ax[1].set(xlabel=r'$\sigma_{J}$', ylabel=r'$\gamma$')
        # ax[1].set(xlabel=r'$\mu_{J}$', ylabel=r'$\gamma$')
        # plt.show()
    


def Jstatepoints(ax, dataset_dictionaries, **pltargs):
    for i, dataset_dictionary in enumerate(dataset_dictionaries):
        file = dataset_dictionary['file']
        group = dataset_dictionary['group']
        dsname = dataset_dictionary['dsname']
        pname = dataset_dictionary['pname']
        hemispheres = ['left', 'right']
        # if i == 0:
        #     for hemisphere in hemispheres:
        #         modpipe = model_pipeline(file, group, dsname)
        #         # modpipe.datasets = np.array(
        #         #     [np.mean(modpipe.datasets, axis=0)])
        #         modpipe.split_hemispheres(which=hemisphere)
        #         _, _, N = modpipe.datasets.shape
        #         J_meanstds = modpipe.select_parameters(pname)
        #         means = J_meanstds[0, :] * N
        #         stds = J_meanstds[1, :] * (N ** 0.5)
        #         ax.plot(means / stds, 1 / stds, **pltargs)
        #         ax.set(
        #             xlabel=r'$\mu _{J} / \sigma_{J}$',
        #             ylabel=r'$1 / \sigma_{J}$')
        #         ax.plot(
        #             np.mean(means) / np.mean(stds),
        #             1 / np.mean(stds), marker='*'
        #             )

        for hemisphere in hemispheres:
            modpipe = model_pipeline(file, group, dsname)
            print(hemisphere, modpipe.datasets.shape)
            modpipe.datasets = np.array(
                [np.mean(modpipe.datasets, axis=0)])
            print(hemisphere, modpipe.datasets.shape)
            modpipe.split_hemispheres(which=hemisphere)
            _, _, N = modpipe.datasets.shape
            J_meanstds = modpipe.select_parameters(pname)
            means = J_meanstds[0, :] * N
            stds = J_meanstds[1, :] * (N ** 0.5)
            ax.plot(means / stds, 1 / stds, **pltargs)
            ax.set(
                xlabel=r'$\mu _{J} / \sigma_{J}$',
                ylabel=r'$1 / \sigma_{J}$')


def Jtails(ax, dataset_dictionaries):
    descriptions = ['averaged', 'grouped']
    for iD, dataset_dictionary in enumerate(dataset_dictionaries):
        file = dataset_dictionary['file']
        group = dataset_dictionary['group']
        dsname = dataset_dictionary['dsname']
        pname = dataset_dictionary['pname']
        hemispheres = ['left', 'right']
        for hemisphere in hemispheres:
            modpipe = model_pipeline(file, group, dsname)
            modpipe.datasets = np.array(
                [np.mean(modpipe.datasets, axis=0)])
            modpipe.split_hemispheres(which=hemisphere)
            _, _, N = modpipe.datasets.shape
            J_meanstdsL = modpipe.select_parameters(pname)
            modpipe.transform_to_distributions(
                    filter_level=0, bins=100, density=True)
            # modpipe.tail_distributions(tail='positive')
            for iMod in range(0, len(modpipe.datasets)):
                xy_tail = modpipe.datasets[iMod]
                # print(xy_tail.shape)
                x = xy_tail[0, :]
                y = xy_tail[1, :]
                ax.plot(
                    x, y,
                    label=descriptions[iD] + '-' +hemisphere)
                ygauss, _ = tools.fit_func1D(tools.gaussian, x, y)
                ax.plot(x, ygauss, c='k', marker=',')

                ax.set(
                    yscale='log',
                    xscale='log',
                    ylim=[1e-3, y.max()])


def Jij_avrg_vs_groupLR(axs, dataset_dictionaries):
    descriptions = ['averaged', 'grouped']
    mods = []
    for iD, dataset_dictionary in enumerate(dataset_dictionaries):
        file = dataset_dictionary['file']
        group = dataset_dictionary['group']
        dsname = dataset_dictionary['dsname']
        pname = dataset_dictionary['pname']
        hemispheres = ['left', 'right']
        for iM, hemisphere in enumerate(hemispheres):
            modpipe = model_pipeline(file, group, dsname)
            modpipe.datasets = np.array(
                [np.mean(modpipe.datasets, axis=0)])
            modpipe.split_hemispheres(which=hemisphere)
            _, _, N = modpipe.datasets.shape
            axs[iD, iM].matshow(modpipe.datasets[0])
            axs[iD, iM].set(title=descriptions[iD] + '-' + hemisphere)
            mods.append(modpipe.datasets[0])
            print(descriptions[iD] + '-' + hemisphere)
    return mods
