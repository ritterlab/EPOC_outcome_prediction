"""
Weighted graph functions taken from the github of Dirk Gütlin
https://gist.github.com/DiGyt/3c06126e678e4b35afdec43a4943917d
"""

import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def dataframe_to_graph_matrix(df):
    corr = df.corr()
    corr = corr.to_numpy()
    matrix = np.copy(corr)
    matrix = np.abs(matrix)
    matrix = remove_self_connections(matrix)
    return(matrix)


def remove_self_connections(matrix):
    """Removes the self-connections of a network graph."""
    for i in range(len(matrix)):
        matrix[i, i] = 0
    return matrix


def invert_matrix(matrix):
    """Invert a matrix from 0 to 1, dependent on whether dealing with distances or weight strengths."""
    new_matrix = 1 - matrix.copy()
    return remove_self_connections(new_matrix)

    
def weighted_shortest_path(matrix):
    """
    Calculate the shortest path lengths between all nodes in a weighted graph.

    This is an implementation of the Floyd-Warshall algorithm for finding the shortest
    path lengths of an entire graph matrix. Implementation taken from:
    https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm

    This implementation is actually identical to scipy.sparse.csgraph.floyd_warshall,
    which we found after the implementation.
    """
    matrix = invert_matrix(matrix)

    n_nodes = len(matrix)
    distances = np.empty([n_nodes, n_nodes])
    for i in range(n_nodes):
        for j in range(n_nodes):
            distances[i,j] = matrix[i, j]
  
    for i in range(n_nodes):
        distances[i,i] = 0

    for k in range(n_nodes):
        for i in range(n_nodes):
            for j in range(n_nodes):
                if distances[i, j] > distances[i, k] + distances[k, j]:
                    distances[i, j] = distances[i, k] + distances[k, j]

    return distances


def weighted_characteristic_path_length(matrix):
    """Calculate the characteristic path length for weighted graphs."""
    n_nodes = len(matrix)
    min_distances = weighted_shortest_path(matrix)

    sum_vector = np.empty(n_nodes)
    for i in range(n_nodes):
        # calculate the inner sum
        sum_vector[i] = (1/(n_nodes-1)) * np.sum([min_distances[i, j] for j in range(n_nodes) if j != i])

    return (1/n_nodes) * np.sum(sum_vector)


def weighted_node_degree(matrix):
    """Calculate the node degree for all nodes in a weighted graph."""
    return np.sum(matrix, axis=-1)


def unweighted_node_degree(matrix):
    """Calculate the node degree for all nodes in a weighted graph."""
    return np.sum(np.ceil(matrix), axis=-1)


def weighted_triangle_number(matrix):
    """Calculate the weighted geometric mean of triangles around i for all nodes i in a weighted graph."""
    n_nodes = len(matrix)

    mean_vector = np.empty([n_nodes])
    for i in range(n_nodes):
        triangles = np.array([[matrix[i, j] * matrix[i, k] * matrix[j, k] for j in range(n_nodes)] for k in range(n_nodes)])**(1/3)
        mean_vector[i] = (1/2) * np.sum(triangles, axis=(0,1))

    return mean_vector


def weighted_clustering_coeff(matrix):
    """Calculate the clustering coefficient for a weighted graph."""
    n = len(matrix)
    t = weighted_triangle_number(matrix)
    k = unweighted_node_degree(matrix) # here we use the !max possible weights as reference
    return (1/n) * np.sum((2 * t)/(k * (k - 1)))


def weighted_clustering_coeff_z(matrix):
    """Zhang's CC is an alternative clustering coefficient which should work better for our case See Samaräki et al. (2008)."""
    n_nodes = len(matrix)
    ccs = []
    for i in range(n_nodes):
        upper = np.sum([[matrix[i,j] * matrix[j,k] * matrix[i,k] for k in range(n_nodes)] for j in range(n_nodes)])
        lower = np.sum([[matrix[i,j] * matrix[i,k] for k in range(n_nodes) if j!=k] for j in range(n_nodes)])
        ccs.append(upper/lower)

    return np.mean(ccs)

def lattice_reference(G, niter=1, D=None, seed=np.random.seed(np.random.randint(0, 2**30))):
    """Latticize the given graph by swapping edges. Works similar to networkx' lattice reference."""
    from networkx.utils import cumulative_distribution, discrete_sequence

    # Instead of choosing uniformly at random from a generated edge list,
    # this algorithm chooses nonuniformly from the set of nodes with
    # probability weighted by degree.
    G = G.copy()
    keys = [i for i in range(len(G))]
    degrees = weighted_node_degree(G)
    cdf = cumulative_distribution(degrees)  # cdf of degree

    nnodes = len(G)
    nedges = nnodes *(nnodes - 1) // 2 # NOTE: assuming full connectivity
    if D is None:
        D = np.zeros((nnodes, nnodes))
        un = np.arange(1, nnodes)
        um = np.arange(nnodes - 1, 0, -1)
        u = np.append((0,), np.where(un < um, un, um))

        for v in range(int(np.ceil(nnodes / 2))):
            D[nnodes - v - 1, :] = np.append(u[v + 1 :], u[: v + 1])
            D[v, :] = D[nnodes - v - 1, :][::-1]

    niter = niter * nedges
    ntries = int(nnodes * nedges / (nnodes * (nnodes - 1) / 2))
    swapcount = 0

    for i in range(niter):
        n = 0
        while n < ntries:
            # pick two random edges without creating edge list
            # choose source node indices from discrete distribution
            (ai, bi, ci, di) = discrete_sequence(4, cdistribution=cdf, seed=seed)
            if len(set((ai, bi, ci, di))) < 4:
                continue  # picked same node twice
            a = keys[ai]  # convert index to label
            b = keys[bi]
            c = keys[ci]
            d = keys[di]

            is_closer = D[ai, bi] >= D[ci, di]
            is_larger = (G[ai, bi] >= G[ci, di])
            if is_closer and is_larger:
                # only swap if we get closer to the diagonal

                ab = G[a, b]
                cd = G[c, d]
                G[a, b] = cd
                G[b, a] = cd
                G[c, d] = ab
                G[d, c] = ab

                swapcount += 1
                break
            n += 1
    return G


def random_reference(G, niter=1, D=None, seed=np.random.seed(np.random.randint(0, 2**30))):
    """Latticize the given graph by swapping edges. Works similar to networkx' random reference."""
    from networkx.utils import cumulative_distribution, discrete_sequence

    # Instead of choosing uniformly at random from a generated edge list,
    # this algorithm chooses nonuniformly from the set of nodes with
    # probability weighted by degree.
    G = G.copy()
    keys = [i for i in range(len(G))]
    degrees = weighted_node_degree(G)
    cdf = cumulative_distribution(degrees)  # cdf of degree

    nnodes = len(G)
    nedges = nnodes *(nnodes - 1) // 2 # NOTE: assuming full connectivity
    if D is None:
        D = np.zeros((nnodes, nnodes))
        un = np.arange(1, nnodes)
        um = np.arange(nnodes - 1, 0, -1)
        u = np.append((0,), np.where(un < um, un, um))

        for v in range(int(np.ceil(nnodes / 2))):
            D[nnodes - v - 1, :] = np.append(u[v + 1 :], u[: v + 1])
            D[v, :] = D[nnodes - v - 1, :][::-1]

    niter = niter * nedges
    ntries = int(nnodes * nedges / (nnodes * (nnodes - 1) / 2))
    swapcount = 0

    for i in range(niter):
        n = 0
        while n < ntries:
            # pick two random edges without creating edge list
            # choose source node indices from discrete distribution
            (ai, bi, ci, di) = discrete_sequence(4, cdistribution=cdf, seed=seed)
            if len(set((ai, bi, ci, di))) < 4:
                continue  # picked same node twice
            a = keys[ai]  # convert index to label
            b = keys[bi]
            c = keys[ci]
            d = keys[di]


            # only swap if we get closer to the diagonal

            ab = G[a, b]
            cd = G[c, d]
            G[a, b] = cd
            G[b, a] = cd
            G[c, d] = ab
            G[d, c] = ab

            swapcount += 1
            break

    return G

def weighted_global_efficiency(matrix):
    """The weighted global efficiency is closely related to the characteristic path length."""
    n_nodes = len(matrix)
    min_distances = weighted_shortest_path(matrix)

    sum_vector = np.empty(n_nodes)
    for i in range(n_nodes):
        # calculate the inner sum
        sum_vector[i] = (1/(n_nodes-1)) * np.sum([1 / min_distances[i, j] for j in range(n_nodes) if j != i])

    return (1/n_nodes) * np.sum(sum_vector)


def weighted_transitivity(matrix):
    """The transitivity is related to the clustering coefficient."""

    n = len(matrix)
    t = weighted_triangle_number(matrix)
    k = unweighted_node_degree(matrix) # here we use the max possible weights as reference

    return np.sum(2 * t) / np.sum(k * (k - 1))

def weighted_sw_sigma(matrix, n_avg=1):
    """Calculate the weighted small world coefficient sigma of a matrix."""
    sigmas = []
    for i in range(n_avg):
        random_graph = random_reference(matrix)
        C = weighted_clustering_coeff_z(matrix)
        C_rand = weighted_clustering_coeff_z(random_graph)
        L = weighted_characteristic_path_length(matrix)
        L_rand = weighted_characteristic_path_length(random_graph)
        sigma = (C/C_rand) / (L/L_rand)
        sigmas.append(sigma)

    return np.mean(sigmas)

def weighted_sw_omega(matrix, n_avg=1):
    """Calculate the weighted small world coefficient omega of a matrix."""
    omegas = []
    for i in range(n_avg):
        random_graph = random_reference(matrix)
        lattice_graph = lattice_reference(matrix)
        C = weighted_clustering_coeff_z(matrix)
        C_latt = weighted_clustering_coeff_z(lattice_graph)
        L = weighted_characteristic_path_length(matrix)
        L_rand = weighted_characteristic_path_length(random_graph)
        omega = (L_rand/L) / (C/C_latt)
        omegas.append(omega)

    return np.mean(omegas)


def weighted_sw_index(matrix, n_avg=1):
    """Calculate the weighted small world coefficient omega of a matrix."""
    indices = []
    for i in range(n_avg):
        random_graph = random_reference(matrix)
        lattice_graph = lattice_reference(matrix)
        C = weighted_clustering_coeff_z(matrix)
        C_rand = weighted_clustering_coeff_z(random_graph)
        C_latt = weighted_clustering_coeff_z(lattice_graph)
        L = weighted_characteristic_path_length(matrix)
        L_rand = weighted_characteristic_path_length(random_graph)
        L_latt = weighted_characteristic_path_length(lattice_graph)
        index = ((L - L_latt) / (L_rand - L_latt)) * ((C - C_rand) / (C_latt - C_rand))
        indices.append(index)
    return np.mean(indices)

def weighted_density(matrix):
    np.fill_diagonal(matrix, 0)
    n_connections = (len(matrix)-1) * len(matrix)
    return(np.sum(matrix)/n_connections)

def get_weighted_graph_statistics(matrix, sw_statistics = True, sw_comparisons = 3):
    """ matrix: a correlation matrix made by dataframe_to_graph_matrix
	sw_statistics: a boolean value whether to calculate the small world statistics
	returns: a dictionary with weighted graph statistics
	Benchmarking: a single SW statistic (sigma, omega) for a 100 node network can take 3-5 minutes on cpus
	The other statistics took significantly less time to calculate """
    features = {}
    features['weighted_transitivity'] = weighted_transitivity(matrix)
    features['weighted_global_efficiency'] = weighted_global_efficiency(matrix)
    features['weighted_clustering_coefficient_zhang'] = weighted_clustering_coeff_z(matrix)
    features['weighted_clustering_coefficient'] = weighted_clustering_coeff(matrix)
    features['weighted_triangle_number'] = np.mean(weighted_triangle_number(matrix))
    features['weighted_density'] = weighted_density(matrix)
    if sw_statistics: 
        features['weighted_sw_sigma'] = weighted_sw_sigma(matrix, sw_comparisons)
        features['weighted_sw_omega'] = weighted_sw_omega(matrix, sw_comparisons)
        features['weighted_sw_omega_2'] = weighted_sw_index(matrix, sw_comparisons)
    return(features)

