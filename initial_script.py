import argparse
import kwant
import matplotlib.pyplot as plt
import numpy as np
import pdb
import scipy.sparse as sp
import time
from graph import GraphReader
from node_stub_matching import stub_matching2
from remapper import Remapper
from scipy.sparse.linalg import eigsh

epsilon = 1e-8
percent_eigenvalue_plot = 10

def stub_matching(graph):
    # low rank approximation: What info do I lose, Ricci flow algorithm (shrink similar edges and expand differents, cut the expanded edges to get clusters), eckart-young-mirsky theorem
    n = len(graph.nodes)
    randomly_ordered_nodes = list(range(1, n+1))
    random.shuffle(randomly_ordered_nodes)
    predicted_edges = []
    predicted_matrix = np.zeros((n, n), dtype=int)
    unmatched_stubs = []
    for node_id in randomly_ordered_nodes:
        used_nodes = {node_id}  # dont allow multi edges
        node = graph.get_node(node_id)
        expected_communities = node.communities
        while node.stubs:
            stub = node.stubs.pop(0)
            initial_nodes = set.intersection(*(graph.communities[comm] for comm in stub))
            initial_nodes -= used_nodes
            probable_nodes = []
            stub_weights = []
            for dest_node_id in initial_nodes:
                dest_node = graph.get_node(dest_node_id)
                stub_counter = dest_node.stubs.count(expected_communities)
                if dest_node.communities == stub and stub_counter > 0:
                    probable_nodes.append(dest_node_id)
                    stub_weights.append(stub_counter)
            if len(probable_nodes) > 0:
                chosen_node_id = random.choices(probable_nodes, weights=stub_weights, k=1)[0]
                chosen_node = graph.get_node(chosen_node_id)
                chosen_node.stubs.remove(expected_communities)
                used_nodes.add(chosen_node_id)
                node.predicted_neighbors.append(chosen_node_id)
                chosen_node.predicted_neighbors.append(node_id)
                predicted_edges.append((min(node_id, chosen_node_id), max(node_id, chosen_node_id)))
                predicted_matrix[node_id-1, chosen_node_id-1] = 1
                predicted_matrix[chosen_node_id-1, node_id-1] = 1
            else:
                unmatched_stubs.append((node_id, stub))
    # Phase 2: Handle unmatched stubs
    if len(unmatched_stubs) > 0:
        print("Going through phase 2...")
    while unmatched_stubs:
        node_id, dest_comms = unmatched_stubs.pop(0)
        node = graph.get_node(node_id)
        dest_nodes = set.intersection(*(graph.communities[comm] for comm in dest_comms))
        dest_nodes = [v for v in dest_nodes if graph.get_node(v).communities == dest_comms and graph.get_node(v).static_stubs.count(node.communities) > 0]
        dest_nodes = list(set(dest_nodes) - set(node.predicted_neighbors) - {node_id})
        # count can be done on open stubs only to reduce execution runtime
        open_stubs = [v for v in dest_nodes if (v, node.communities) in unmatched_stubs and graph.get_node(v).static_stubs.count(node.communities) > 0]
        if len(open_stubs) > 0:
            chosen_node_id = random.choice(open_stubs)
            predicted_edges.append((min(node_id, chosen_node_id), max(node_id, chosen_node_id)))
            predicted_matrix[node_id-1, chosen_node_id-1] = 1
            predicted_matrix[chosen_node_id-1, node_id-1] = 1
            chosen_node = graph.get_node(chosen_node_id)
            chosen_node.predicted_neighbors.append(node_id)
            node.predicted_neighbors.append(chosen_node_id)
            unmatched_stubs.remove((chosen_node_id, node.communities))
        else:
            # Can be further optimized by solving all issues with this community combination pair as we already have the subgraph from the two community combinations
            src_nodes = set.intersection(*(graph.communities[comm] for comm in node.communities)) - {node_id}
            src_nodes = [v for v in src_nodes if graph.get_node(v).communities == node.communities and graph.get_node(v).static_stubs.count(dest_comms) > 0]
            chosen_node_id = random.choice(dest_nodes)
            predicted_edges.append((min(node_id, chosen_node_id), max(node_id, chosen_node_id)))
            predicted_matrix[node_id-1, chosen_node_id-1] = 1
            predicted_matrix[chosen_node_id-1, node_id-1] = 1
            chosen_node = graph.get_node(chosen_node_id)
            chosen_node.predicted_neighbors.append(node_id)
            node.predicted_neighbors.append(chosen_node_id)
            src_nodes = [v for v in src_nodes if chosen_node_id in graph.get_node(v).predicted_neighbors]
            broken_node_id = random.choice(src_nodes)
            broken_node = graph.get_node(broken_node_id)
            broken_node.predicted_neighbors.remove(chosen_node_id)
            chosen_node.predicted_neighbors.remove(broken_node_id)
            predicted_edges.remove((min(chosen_node_id, broken_node_id), max(chosen_node_id, broken_node_id)))
            predicted_matrix[chosen_node_id-1, broken_node_id-1] = 0
            predicted_matrix[broken_node_id-1, chosen_node_id-1] = 0
            unmatched_stubs.append((broken_node_id, dest_comms))

    return predicted_edges, predicted_matrix

def plot_spectral_density(adjacency_matrix, ax=None):
    degrees = np.array(adjacency_matrix.sum(axis=1)).flatten()
    degree_matrix = sp.diags(degrees)
    laplacian_matrix = degree_matrix - adjacency_matrix
    L = sp.csr_matrix(laplacian_matrix)
    dos = kwant.kpm.SpectralDensity(L)
    k = max(2, int(percent_eigenvalue_plot * 0.01 * adjacency_matrix.shape[0]))
    eigenvalues = eigsh(L, k=k, return_eigenvectors=False)
#    eigenvalues = eigsh(L, k=k, return_eigenvectors=False, which="SM")
    padding = 0.1 * (eigenvalues.max() - eigenvalues.min())
    energy_range = (eigenvalues.min() - padding, eigenvalues.max() + padding)
    energies = np.linspace(energy_range[0], energy_range[1], 200)
    rho = dos(energies)
    if ax is None:
        ax = plt.gca()
    ax.plot(energies, rho)
    ax.set_xlabel("Eigenvalues")
    ax.set_ylabel("Density")
    ax.set_title("Eigenvalue Distribution of Graph Laplacian (KPM)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--community_size",
        type=int,
        default=10,
        help="Number of communities in the graph (default: %(default)s)"
    )
    parser.add_argument(
        "-d", "--child_directory",
        type=str,
        default="dblp",
        help="Dataset you want to work with (default: %(default)s)"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-g", "--generate_data",
        action="store_true",
        help="Generate smaller dataset"
    )
    group.add_argument(
        "-e", "--execute_algorithm",
        action="store_true",
        help="Execute algorithm on the dataset"
    )
    args = parser.parse_args()
    if args.generate_data:
        remapper = Remapper(args.community_size, args.child_directory)
        remapper.process()
    else:
        reader = GraphReader(args.community_size, args.child_directory)
        graph = reader.read_metis_file()
        start_time = time.time()
        predicted_edges1, predicted_matrix1, predicted_edges2, predicted_matrix2 = stub_matching2(graph)
        end_time = time.time()

        num_nodes = len(graph.nodes)
        original_edges = set()
        for i in range(num_nodes):
            for j in range(i, num_nodes):
                if graph.adjacency_matrix[i, j] == 1:
                    original_edges.add((i+1, j+1))
        print(f"Time taken: {end_time - start_time} seconds. Newly seen edges:: Random: {(len(set(predicted_edges1) - original_edges)*100.0)/len(predicted_edges1)}% Careful: {(len(set(predicted_edges2) - original_edges)*100.0)/len(predicted_edges2)}%")

        fig, axes = plt.subplots(1, 3, figsize=(16, 8))
        plot_spectral_density(graph.adjacency_matrix, ax=axes[0])
        plot_spectral_density(predicted_matrix1, ax=axes[1])
        plot_spectral_density(predicted_matrix2, ax=axes[2])
        ymin = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0], axes[2].get_ylim()[0])
        ymax = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1], axes[2].get_ylim()[1])
        axes[0].set_ylim(ymin, ymax)
        axes[1].set_ylim(ymin, ymax)
        axes[2].set_ylim(ymin, ymax)
        plt.tight_layout()
        plt.show()
#        plt.savefig(f"dataset/{args.child_directory}/small_graphs/plot{args.community_size}.png")
#        plt.close()

