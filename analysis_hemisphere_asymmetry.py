import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from pyplm.pipelines import model_pipeline
from pyplm.analyse import tail
from pyplm.utilities import tools

plt.style.use("/Users/mk14423/Dropbox/custom.mplstyle")

file = '/Users/mk14423/Desktop/PaperData/HCP_data_Ryota/HCP_rsfmri.hdf5'
group = 'individuals'
dataset = 'inferredModels'
pname='J'

def statepoints_overview(N = 180):
    # SK: y=kT/Jstd 
    modpipeL = model_pipeline(file, group, dataset)
    modpipeL.split_hemispheres(which='left')
    J_meanstdsL = modpipeL.select_parameters(pname)
    # modpipeL.transform_to_distributions(0, bins=100, density=True)
    # modpipeL.tail_distributions(tail='positive')
    # mcsL = tail.fit_power_laws(modpipeL.datasets)

    modpipeR = model_pipeline(file, group, dataset)
    modpipeR.split_hemispheres(which='right')
    J_meanstdsR = modpipeR.select_parameters(pname)
    # modpipeR.transform_to_distributions(0, bins=100, density=True)
    # modpipeR.tail_distributions(tail='positive')
    # mcsR = tail.fit_power_laws(modpipeR.datasets)

    left_means = J_meanstdsL[0, :] * N
    left_stds = J_meanstdsL[1, :] * (N ** 0.5)
    right_means = J_meanstdsR[0, :] * N
    right_stds = J_meanstdsR[1, :] * (N ** 0.5)
    # fig, ax = plt.subplots(2, 1, figsize=(6,5))
    # ax = ax.ravel()
    fig, ax = plt.subplots(figsize=(6,5))
    ax = [ax]
    # connect two of the same thing with a line?
    # then this defines some error distance
    # in two2 space?
    ax[0].plot(left_means / left_stds, 1 / left_stds, ls='none')
    ax[0].plot(right_means / right_stds, 1 / right_stds, ls='none')
    for iROI in range(0, len(left_stds)):
        ax[0].plot(
            [left_means / left_stds, right_means / right_stds],
            [1 / left_stds, 1 / right_stds],
            c='k', marker=',', alpha=0.5
            )
    ax[0].set(
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

    plt.show()

def statepoints_assymetry(N=180):
    modpipeL = model_pipeline(file, group, dataset)
    modpipeL.split_hemispheres(which='left')
    J_meanstdsL = modpipeL.select_parameters(pname)
    # modpipeL.transform_to_distributions(0, bins=100, density=True)
    # modpipeL.tail_distributions(tail='positive')
    # mcsL = tail.fit_power_laws(modpipeL.datasets)

    modpipeR = model_pipeline(file, group, dataset)
    modpipeR.split_hemispheres(which='right')
    J_meanstdsR = modpipeR.select_parameters(pname)
    # modpipeR.transform_to_distributions(0, bins=100, density=True)
    # modpipeR.tail_distributions(tail='positive')
    # mcsR = tail.fit_power_laws(modpipeR.datasets)

    J_meanstdsL[0, :] = J_meanstdsL[0, :] * N
    J_meanstdsL[1, :] = J_meanstdsL[1, :] * (N ** 0.5)
    J_meanstdsR[0, :] = J_meanstdsR[0, :] * N
    J_meanstdsR[1, :] = J_meanstdsR[1, :] * (N ** 0.5)

    left_means = J_meanstdsL[0, :]
    left_stds = J_meanstdsL[1, :]
    right_means = J_meanstdsR[0, :]
    right_stds = J_meanstdsR[1, :]
    # euclidain distance?
    dist_both = np.linalg.norm(J_meanstdsL - J_meanstdsR, axis=0)
    dist_means = np.abs(left_means - right_means)
    dist_stds = np.abs(left_stds - right_stds)
    print(np.argsort(dist_both)[-10:])
    print(np.argsort(dist_means)[-10:])
    print(np.argsort(dist_stds)[-10:])
    print(dist_both.shape, dist_means.shape)
    # dist_means = left_means - right_means # <0 is right bigger
    # dist_stds = left_stds - right_stds # >0 is left bigger
    fig, ax = plt.subplots(3, 1)
    ax = ax.ravel()
    ax[0].hist(dist_both, bins=20)
    ax[1].hist(dist_means, bins=20)
    ax[2].hist(dist_stds, bins=20)
    ax[0].set(xlabel=r'$\Delta \mu_{J}$')
    ax[1].set(xlabel=r'$\Delta \sigma_{J}$')
    plt.tight_layout(pad=0)
    plt.show()


def individuals_assymetry(iIndividuals=[7, 13, 127], N=180):
    modpipeL = model_pipeline(file, group, dataset)
    modpipeL.split_hemispheres(which='left')
    # modelsL = np.array([modpipeL.datasets[i] for i in iIndividuals])
    modelsL = np.copy(modpipeL.datasets)
    J_meanstdsL = modpipeL.select_parameters(pname)
    # modpipeL.transform_to_distributions(0, bins=100, density=True)
    # modpipeL.tail_distributions(tail='positive')
    # mcsL = tail.fit_power_laws(modpipeL.datasets)

    modpipeR = model_pipeline(file, group, dataset)
    modpipeR.split_hemispheres(which='right')
    # modelsR = np.array([modpipeR.datasets[i] for i in iIndividuals])
    modelsR = np.copy(modpipeR.datasets)
    J_meanstdsR = modpipeR.select_parameters(pname)

    matrix_distance = np.linalg.norm(modelsL - modelsR, axis=(1,2))
    print(matrix_distance.shape)
    print(np.argsort(matrix_distance)[-10:])
    # for i in range(0, len(iIndividuals)):
    #     fig, ax = plt.subplots(1, 2)
    #     ax = ax.ravel()
    #     ax[0].imshow(modelsL[i])
    #     ax[1].imshow(modelsR[i])
    #     plt.tight_layout(pad=0)
    #     plt.show()
    
    # # modpipeR.transform_to_distributions(0, bins=100, density=True)
    # # modpipeR.tail_distributions(tail='positive')
    # # mcsR = tail.fit_power_laws(modpipeR.datasets)

    # J_meanstdsL[0, :] = J_meanstdsL[0, :] * N
    # J_meanstdsL[1, :] = J_meanstdsL[1, :] * (N ** 0.5)
    # J_meanstdsR[0, :] = J_meanstdsR[0, :] * N
    # J_meanstdsR[1, :] = J_meanstdsR[1, :] * (N ** 0.5)

    # left_means = J_meanstdsL[0, :]
    # left_stds = J_meanstdsL[1, :]
    # right_means = J_meanstdsR[0, :]
    # right_stds = J_meanstdsR[1, :]
    # # euclidain distance?
    # dist_both = np.linalg.norm(J_meanstdsL - J_meanstdsR, axis=0)
    # dist_means = np.abs(left_means - right_means)
    # dist_stds = np.abs(left_stds - right_stds)
    # print(np.argsort(dist_both)[-10:])
    # print(np.argsort(dist_means)[-10:])
    # print(np.argsort(dist_stds)[-10:])
    # print(dist_both.shape, dist_means.shape)
    # # dist_means = left_means - right_means # <0 is right bigger
    # # dist_stds = left_stds - right_stds # >0 is left bigger
    # fig, ax = plt.subplots(3, 1)
    # ax = ax.ravel()
    # ax[0].hist(dist_both, bins=20)
    # ax[1].hist(dist_means, bins=20)
    # ax[2].hist(dist_stds, bins=20)
    # ax[0].set(xlabel=r'$\Delta \mu_{J}$')
    # ax[1].set(xlabel=r'$\Delta \sigma_{J}$')
    # plt.tight_layout(pad=0)
    # plt.show()

def get_Aij(which_hemisphere, iInvidividals=None):
    modpipe = model_pipeline(file, group, dataset)
    modpipe.split_hemispheres(which=which_hemisphere)
    if iInvidividals is not None:
        modpipe.datasets = np.array([modpipe.datasets[i] for i in iInvidividals])
    models = np.copy(modpipe.datasets)
    modpipe.select_parameters(pname)
    modpipe.transform_to_distributions(0, bins=100, density=True)
    _, thresholds = modpipe.tail_distributions(tail='positive')
    Aijs = []
    for i in range(0, len(thresholds)):
        th = thresholds[i]
        model = np.copy(models[i])
        model[model <= th] = 0
        model[model > th] = 1
        Aijs.append(model)
    Aijs = np.array(Aijs)
    return Aijs

def individuals_Aij_assymetry(iIndividuals=[7, 13, 127], N=180):
    modpipe = model_pipeline(file, group, dataset)
    modelsW = modpipe.datasets
    # modpipe.split_hemispheres(which='left')
    # modelsL = modpipe.datasets
    # modpipe = model_pipeline(file, group, dataset)
    # modpipe.split_hemispheres(which='right')
    # modelsR = modpipe.datasets
    # I think it makes more sense if I define a similarity
    # not a correlation?
    modelsL = get_Aij('left', iIndividuals)
    modelsR = get_Aij('right', iIndividuals)
    connectivityL = np.array([tools.triu_flat(mod, k=0) for mod in modelsL])
    connectivityR = np.array([tools.triu_flat(mod, k=0) for mod in modelsR])
    connectivityW = np.array([tools.triu_flat(mod, k=0) for mod in modelsW])
    nSubjects, nSamples = connectivityL.shape
    connectivity = np.vstack((connectivityL, connectivityR))
    # connectivity = connectivityW
    # what am I plotting here lol?
    # something about summing along to show something?
    print(connectivity.shape)
    subject_matrix = np.corrcoef(connectivity)
    # dims are y, x for matshow!
    sm_LL = subject_matrix[0:161, 0:161]
    sm_RR = subject_matrix[161:, 161:]
    sm_LR = subject_matrix[161:, 0:161]
    # sm_RL = subject_matrix[0:161, 161:]
    # print(np.allclose(sm_LR, sm_RL.T))
    # these are correlations!
    sms = [sm_LL, sm_RR, sm_LR]
    np.fill_diagonal(sm_LL, 0)
    np.fill_diagonal(sm_RR, 0)
    LR_sim = np.diag(sm_LR)
    LL_sim = [
        np.sum(sm_LL[i])/(len(sm_LL)-1) for i in range(0, len(sm_LL))]
    RR_sim = [
        np.sum(sm_RR[i])/(len(sm_RR)-1) for i in range(0, len(sm_RR))]
    # define some other similarty at the moment it's correlations?
    # let's get an adjaceny, then a dfiference between adjacencies
    # matrix for that will do!
    # LL_sim = np.mean(sm_LL, axis=0)
    # RR_sim = np.mean(sm_RR, axis=0)
    # LL_std = np.std(sm_LL, axis=0)
    # RR_std = np.std(sm_RR, axis=0)
    # plt.plot(LL_sim, LL_std)
    # plt.show()
    plt.plot(LL_sim, LL_sim, c='k', marker=',')
    plt.plot(
        LL_sim, RR_sim, ls='none',
        label=f'r2 = {r2_score(LL_sim, RR_sim):.3f}')
    plt.legend()
    plt.show()

    # calculate what the mean is and do a differnece
    # from the mean?
    plt.errorbar(
        x=np.arange(0, 161),
        y=LL_sim - np.mean(LL_sim),
        # yerr=LL_std
        )
    plt.axhline(np.std(LL_sim), c='k', marker=',')
    plt.axhline(- np.std(LL_sim), c='k', marker=',')
    plt.errorbar(
        x=np.arange(0, 161),
        y=RR_sim - np.mean(LL_sim),
        # yerr=RR_std
        )
    plt.plot(LR_sim - np.mean(LR_sim))
    # plt.plot(LL_similarity, LL_similarity, ls='none', c='k')
    # plt.plot(LL_similarity, RR_similarity, ls='none')
    # plt.plot(LL_similarity, LR_similarity, ls='none')
    plt.show()
    exit()
    # so I think the only really meaningful thing is the diagonal of sm_LR?
    # how much does L correlate with R of subject i?
    # summing along the axis tells me something, just not sure
    # what that something is. Or averaging across the axis?
    # the diagonal of LR tells me how similar the two halfs are!
    fig, ax = plt.subplots(1, 3)
    ax = ax.ravel()
    for i in range(0, len(sms)):
        # np.fill_diagonal(sms[i], 0)
        ax[i].matshow(sms[i])  #  [0:20, 0:20]
    plt.show()
    plt.imshow(subject_matrix)
    plt.show()
    exit()
    fig, ax = plt.subplots(1, 2)
    i=0
    ax = ax.ravel()
    ax[0].imshow(modelsL[i])
    ax[1].imshow(modelsR[i])
    plt.tight_layout(pad=0)
    plt.show()
    # covariance to get how similar they are?

    matrix_distance = np.linalg.norm(modelsL - modelsR, axis=(1,2))
    print(matrix_distance.shape)
    plt.hist(matrix_distance, bins=20)
    plt.show()
    print(np.argsort(matrix_distance)[-10:])
    # let's do a matrix which plots how similar each of the other matricies are
    # i.e. an error across subjects, with the distances?
    # the matricies are symmetric so x2 of what the real distance is
    # for i in range(0, len(modelsL)):
    #     fig, ax = plt.subplots(1, 2)
    #     ax = ax.ravel()
    #     ax[0].imshow(modelsL[i])
    #     ax[1].imshow(modelsR[i])
    #     plt.tight_layout(pad=0)
    #     plt.show()


def individuals_Aij_assymetry2(iIndividuals=[7, 13, 127], N=180):
    # not sure this means much
    # modelsL = get_Aij('left', iIndividuals)
    # modelsR = get_Aij('right', iIndividuals)
    modpipe = model_pipeline(file, group, dataset)
    modpipe.split_hemispheres(which='left')
    modelsL = modpipe.datasets
    modpipe = model_pipeline(file, group, dataset)
    modpipe.split_hemispheres(which='right')
    modelsR = modpipe.datasets
    # I think it makes more sense if I define a similarity
    # not a correlation?
    nRois = 180
    nParams = (nRois * (nRois-1)) / 2
    connectivityL = np.array([tools.triu_flat(mod, k=0) for mod in modelsL])
    connectivityR = np.array([tools.triu_flat(mod, k=0) for mod in modelsR])
    LR_difference = modelsL - modelsR
    LR_conDif = connectivityL - connectivityR
    difference_difference = np.corrcoef(LR_conDif)
    np.fill_diagonal(difference_difference, 0)
    print(LR_conDif.shape, difference_difference.shape)
    plt.matshow(difference_difference)
    plt.show()
    # this deosn't take into account if same differences
    # so should do something else with it now to see if same
    # ROIs identified across all?
    # something convolve?
    # yep I want the difference in the differences!
    connection_assymetry = [
        np.count_nonzero(LR_difference[i]) / nParams
        for i in range(0, len(LR_difference))]
    print(LR_difference.shape)
    fig, ax = plt.subplots(1, 3)
    ax = ax.ravel()
    for i in range(0, 3):
        ax[i].matshow(LR_difference[i])


    plt.matshow(LR_difference[1]-LR_difference[0])
    plt.show()

    print(np.count_nonzero(LR_difference[0]) / nParams)
    plt.show()
    plt.plot(connection_assymetry)
    plt.show()
    # distanceLR = np.linalg.norm(connectivityL - connectivityR, axis=(1))
    # distanceLR2 = np.linalg.norm(modelsR - modelsL, axis=(1,2))
    # print(distanceLR.shape)
    # print(distanceLR2/distanceLR)
    # plt.plot(distanceLR)
    # plt.plot(distanceLR2)
    # plt.show()
    exit()
    nSubjects, nSamples = connectivityL.shape
    connectivity = np.vstack((connectivityL, connectivityR))
    # connectivity = connectivityW
    # what am I plotting here lol?
    # something about summing along to show something?
    print(connectivity.shape)
    subject_matrix = np.corrcoef(connectivity)
    # dims are y, x for matshow!
    sm_LL = subject_matrix[0:161, 0:161]
    sm_RR = subject_matrix[161:, 161:]
    sm_LR = subject_matrix[161:, 0:161]
    # sm_RL = subject_matrix[0:161, 161:]
    # print(np.allclose(sm_LR, sm_RL.T))
    # these are correlations!
    sms = [sm_LL, sm_RR, sm_LR]
    np.fill_diagonal(sm_LL, 0)
    np.fill_diagonal(sm_RR, 0)
    LR_sim = np.diag(sm_LR)
    LL_sim = [
        np.sum(sm_LL[i])/(len(sm_LL)-1) for i in range(0, len(sm_LL))]
    RR_sim = [
        np.sum(sm_RR[i])/(len(sm_RR)-1) for i in range(0, len(sm_RR))]
    # define some other similarty at the moment it's correlations?
    # let's get an adjaceny, then a dfiference between adjacencies
    # matrix for that will do!
    # LL_sim = np.mean(sm_LL, axis=0)
    # RR_sim = np.mean(sm_RR, axis=0)
    # LL_std = np.std(sm_LL, axis=0)
    # RR_std = np.std(sm_RR, axis=0)
    # plt.plot(LL_sim, LL_std)
    # plt.show()
    plt.plot(LL_sim, LL_sim, c='k', marker=',')
    plt.plot(
        LL_sim, RR_sim, ls='none',
        label=f'r2 = {r2_score(LL_sim, RR_sim):.3f}')
    plt.legend()
    plt.show()

    # calculate what the mean is and do a differnece
    # from the mean?
    plt.errorbar(
        x=np.arange(0, 161),
        y=LL_sim - np.mean(LL_sim),
        # yerr=LL_std
        )
    plt.axhline(np.std(LL_sim), c='k', marker=',')
    plt.axhline(- np.std(LL_sim), c='k', marker=',')
    plt.errorbar(
        x=np.arange(0, 161),
        y=RR_sim - np.mean(LL_sim),
        # yerr=RR_std
        )
    plt.plot(LR_sim - np.mean(LR_sim))
    # plt.plot(LL_similarity, LL_similarity, ls='none', c='k')
    # plt.plot(LL_similarity, RR_similarity, ls='none')
    # plt.plot(LL_similarity, LR_similarity, ls='none')
    plt.show()
    exit()
    # so I think the only really meaningful thing is the diagonal of sm_LR?
    # how much does L correlate with R of subject i?
    # summing along the axis tells me something, just not sure
    # what that something is. Or averaging across the axis?
    # the diagonal of LR tells me how similar the two halfs are!
    fig, ax = plt.subplots(1, 3)
    ax = ax.ravel()
    for i in range(0, len(sms)):
        # np.fill_diagonal(sms[i], 0)
        ax[i].matshow(sms[i])  #  [0:20, 0:20]
    plt.show()
    plt.imshow(subject_matrix)
    plt.show()
    exit()
    fig, ax = plt.subplots(1, 2)
    i=0
    ax = ax.ravel()
    ax[0].imshow(modelsL[i])
    ax[1].imshow(modelsR[i])
    plt.tight_layout(pad=0)
    plt.show()
    # covariance to get how similar they are?

    matrix_distance = np.linalg.norm(modelsL - modelsR, axis=(1,2))
    print(matrix_distance.shape)
    plt.hist(matrix_distance, bins=20)
    plt.show()
    print(np.argsort(matrix_distance)[-10:])
    # let's do a matrix which plots how similar each of the other matricies are
    # i.e. an error across subjects, with the distances?
    # the matricies are symmetric so x2 of what the real distance is
    # for i in range(0, len(modelsL)):
    #     fig, ax = plt.subplots(1, 2)
    #     ax = ax.ravel()
    #     ax[0].imshow(modelsL[i])
    #     ax[1].imshow(modelsR[i])
    #     plt.tight_layout(pad=0)
    #     plt.show()

def individuals_crossIndividual_Similarity():
    # not sure this means much
    # modelsL = get_Aij('left', iIndividuals)
    # modelsR = get_Aij('right', iIndividuals)
    modpipe = model_pipeline(file, group, dataset)
    models = modpipe.datasets
    connectivity = np.array([tools.triu_flat(mod, k=0) for mod in models])
    
    subjectCorrs = np.corrcoef(connectivity)
    subjectCorrs = subjectCorrs  # [0:10, 0:10]
    subjectCorrs_flat = tools.triu_flat(subjectCorrs, k=1)
    vmax = subjectCorrs_flat.max()
    vmin = subjectCorrs_flat.min()
    np.fill_diagonal(subjectCorrs, 0)

    subjectCorrs_avrgs = [
        np.sum(subjectCorrs[i])/(len(subjectCorrs)-1)
        for i in range(0, len(subjectCorrs))]
    args = np.argsort(subjectCorrs_avrgs)
    print(args)
    sortedCorrs = np.zeros_like(subjectCorrs)
    for i, arg in enumerate(args):
        sortedCorrs[i, :] = subjectCorrs[arg, :]

    fig, ax = plt.subplots(1, 2)
    ax = ax.ravel()
    ax[0].matshow(subjectCorrs, vmin=vmin, vmax=vmax)
    ax[1].matshow(sortedCorrs, vmin=vmin, vmax=vmax)

    # plt.colorbar()
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(np.sort(subjectCorrs_avrgs))
    plt.show()
    exit()
    nSubjects, nSamples = connectivityL.shape
    connectivity = np.vstack((connectivityL, connectivityR))
    # connectivity = connectivityW
    # what am I plotting here lol?
    # something about summing along to show something?
    print(connectivity.shape)
    subject_matrix = np.corrcoef(connectivity)
    # dims are y, x for matshow!
    sm_LL = subject_matrix[0:161, 0:161]
    sm_RR = subject_matrix[161:, 161:]
    sm_LR = subject_matrix[161:, 0:161]
    # sm_RL = subject_matrix[0:161, 161:]
    # print(np.allclose(sm_LR, sm_RL.T))
    # these are correlations!
    sms = [sm_LL, sm_RR, sm_LR]
    np.fill_diagonal(sm_LL, 0)
    np.fill_diagonal(sm_RR, 0)
    LR_sim = np.diag(sm_LR)
    LL_sim = [
        np.sum(sm_LL[i])/(len(sm_LL)-1) for i in range(0, len(sm_LL))]
    RR_sim = [
        np.sum(sm_RR[i])/(len(sm_RR)-1) for i in range(0, len(sm_RR))]
    # define some other similarty at the moment it's correlations?
    # let's get an adjaceny, then a dfiference between adjacencies
    # matrix for that will do!
    # LL_sim = np.mean(sm_LL, axis=0)
    # RR_sim = np.mean(sm_RR, axis=0)
    # LL_std = np.std(sm_LL, axis=0)
    # RR_std = np.std(sm_RR, axis=0)
    # plt.plot(LL_sim, LL_std)
    # plt.show()
    plt.plot(LL_sim, LL_sim, c='k', marker=',')
    plt.plot(
        LL_sim, RR_sim, ls='none',
        label=f'r2 = {r2_score(LL_sim, RR_sim):.3f}')
    plt.legend()
    plt.show()


statepoints_overview()
statepoints_assymetry()
# individuals_assymetry()
# individuals_Aij_assymetry2(iIndividuals=None)
# individuals_crossIndividual_Similarity()
