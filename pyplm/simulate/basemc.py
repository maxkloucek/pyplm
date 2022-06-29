import numpy as np
from numba import njit
from numba.typed import List

def simulate(interaction_matrix, initial_config, mc_cycles, cycle_dumpfreq=10):

    initial_energy = calc_energy(initial_config, interaction_matrix)

    neighbour_list = build_typed_neighbour_list(
                interaction_matrix, TH=0)

    # _, state, _, _, _ = np.random.get_state()
    # print('{}'.format(state[-10:-5]))
    trajectory, energy = sim(
        interaction_matrix, initial_config, initial_energy,
        neighbour_list, mc_cycles, cycle_dumpfreq)
    trajectory = np.array(trajectory)
    return trajectory, energy


def calc_energy(configuration, interaction_matrix):
    N = configuration.size
    # neighbour_list = mc.build_typed_neighbour_list(
    #    interaction_matrix, d=2, TH=0)

    upper_indices = np.triu_indices(N, k=1)
    interactions = interaction_matrix[upper_indices]
    config_matrix = np.outer(configuration, configuration)
    config_pairs = config_matrix[upper_indices]  # gives i<j terms
    E = - np.sum(interactions * config_pairs) - \
        np.sum(np.diagonal(interaction_matrix)*configuration)
    return E


@njit
def sim(
        int_matrix, init_config, init_energy, neighbour_list,
        mc_cycles, cycle_dumpfreq):
    # print('---NJIT LOOP START---')
    T = 1  # do it explictly for now
    nParticles = init_config.size
    config = np.copy(init_config)
    E = init_energy

    nSamples = int(mc_cycles / cycle_dumpfreq)
    nSteps_per_sample = int(nParticles * cycle_dumpfreq)
    configs = np.empty((nSamples, nParticles))  # , dtype=np.intc?
    energy = np.empty((nSamples))

    for sampleIndex in range(0, nSamples):
        rand_nos = np.random.random(nSteps_per_sample)
        trial_indicies = np.random.randint(0, nParticles, nSteps_per_sample)

        for mc_step in range(0, nSteps_per_sample):
            trial_index = trial_indicies[mc_step]
            dE = 0
            s_old = config[trial_index]
            s_new = -config[trial_index]
            ds = s_new - s_old

            connected_indices = neighbour_list[trial_index]
            for j in connected_indices:
                dE = dE - (int_matrix[trial_index, j] * config[j] * ds)
            # I think I had this missing for the diagonal!!
            dE = dE - s_new*int_matrix[trial_index, trial_index]
            if (dE / T) < 0:
                config[trial_index] = -config[trial_index]
                E = E + dE
            else:
                if np.exp(-(dE / T)) >= rand_nos[mc_step]:
                    config[trial_index] = -config[trial_index]
                    E = E + dE
        configs[sampleIndex, :] = np.copy(config)
        energy[sampleIndex] = E
    return configs, energy


def build_neighbour_list(interaction_matrix, TH):
    # this is to make sure self isn't included in neighbour list keeping,
    # as self interaction is different to pair interaction!
    # looks like it doesnt really always work!
    no_particles, _ = interaction_matrix.shape
    J_neighbours = np.copy(interaction_matrix)
    np.fill_diagonal(J_neighbours, 0)
    # does this work for h=!0? Need to check & be careful I think?
    nonzero_list = np.argwhere(np.abs(J_neighbours) > TH)
    neighbour_list = [[] for _ in range(no_particles)]
    neighbour_length, _ = nonzero_list.shape
    for i in range(0, neighbour_length):
        list_index = nonzero_list[i, 0]
        neighbour_list[list_index].append(nonzero_list[i, 1])
    return neighbour_list


def build_typed_neighbour_list(interaction_matrix, TH):
    neighbour_list = build_neighbour_list(interaction_matrix, TH)
    typed_neighbour_list = List()
    for i in range(len(neighbour_list)):
        new_list = List()
        for j in range(len(neighbour_list[i])):
            new_list.append(neighbour_list[i][j])
        typed_neighbour_list.append(new_list)
    return typed_neighbour_list


# return 1D object containing list of initial Ising Spins
# (i.e. +/- 1 only -> binary)
# this should go to a different function block!
def initialise_ising_config(N, option):
    if option == -1:
        config = -np.ones(N)
    elif option == 0:
        config = np.random.randint(2, size=N)
        config[config == 0] = -1
    elif option == 1:
        config = np.ones(N)
    else:
        print('Invalid initialisation choice made')
        return 1
    return config

