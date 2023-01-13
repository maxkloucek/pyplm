import numpy as np
import matplotlib.pyplot as plt


def mkfigure(**subplotkwargs):
    fig, ax = plt.subplots(squeeze=False, **subplotkwargs)
    labels = [
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
        'n', 'o', 'p', 'q', 'r', 't', 'u', 'v'
        ]
    labels = [letter + ')' for letter in labels]
    labels = ['(' + letter for letter in labels]
    # labels[0] = labels[0] + ' N=200'
    # labels[1] = labels[1] + ' N=400'
    if ax.size != 1:
        ax_ravel = ax.ravel()
        for iax in range(0, ax_ravel.size):
            # ax_ravel[iax].text(
            #     1.0, 1.0, labels[iax], transform=ax_ravel[iax].transAxes,
            #     # fontsize='medium', fontfamily='serif',
            #     horizontalalignment='right',
            #     verticalalignment='top',
            #     bbox=dict(facecolor='0.7', edgecolor='none', pad=0))
            ax_ravel[iax].text(
                0.0, 1.0, labels[iax], transform=ax_ravel[iax].transAxes,
                # fontsize='medium', fontfamily='serif',
                horizontalalignment='left',
                verticalalignment='top',
                bbox=dict(facecolor='0.7', edgecolor='none', pad=0))
    return fig, ax


def add_subplot_labels(axs):
    labels = [
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
        'n', 'o', 'p', 'q', 'r', 't', 'u', 'v'
        ]
    labels = [letter + ')' for letter in labels]
    labels = ['(' + letter for letter in labels]

    axs = axs.ravel()
    for iax in range(0, axs.size):
        axs[iax].text(
            0.0, 1.0, labels[iax], transform=axs[iax].transAxes,
            # fontsize='medium', fontfamily='serif',
            horizontalalignment='left',
            verticalalignment='top',
            bbox=dict(facecolor='0.7', edgecolor='none', pad=0))


def category_col(iC):
    CAT_COLS = [
        '#4053d3', '#ddb310', '#b51d14',
        '#00beff', '#fb49b0', '#00b25d', '#cacaca']
    return CAT_COLS[iC]


def histogram(ax, parameters, nbins, invert=False, **pltargs):
    n, bin_edges = np.histogram(parameters, bins=nbins, density=True)
    bin_centers = bin_edges[:-1] + 0.5 * (bin_edges[1] - bin_edges[0])
    bin_centers = bin_centers[n > 0]
    n = n[n > 0]
    # # decide on this later...
    # this only applies if equally spaced?
    n = n * (bin_edges[1] - bin_edges[0])
    ax.plot(
        bin_centers, n,
        # marker=',',
        linewidth=2,
        **pltargs,
    )
    print(f'prob_sum = {np.sum(n):.3}')
    if invert is True:
        ax.invert_xaxis()

    return n, bin_centers
