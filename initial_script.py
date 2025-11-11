import argparse
import kwant
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pdb
import random
import scipy.sparse as sp
import time
from collections import defaultdict
from matplotlib.patches import Patch, Ellipse
from scipy.sparse.linalg import eigsh

epsilon = 1e-8

class Remapper:
    edge_file: str
    comm_file: str
    new_edge_file: str
    new_comm_file: str
    mapping: dict[int, int]
    community_size: int

    def __init__(self, community_size=10, child_directory="dblp"):
        self.edge_file = f"dataset/{child_directory}/com-{child_directory}.ungraph.txt"
        self.comm_file = f"dataset/{child_directory}/com-{child_directory}.top5000.cmty.txt"
        self.new_edge_file = f"dataset/{child_directory}/small_graphs/graph{community_size}.metis"
        self.new_comm_file = f"dataset/{child_directory}/small_graphs/communities{community_size}.txt"
        self.mapping = {}
        self.community_size = community_size

    def build_mapping(self):
        nodes = set()
        with open(self.edge_file, "r") as f:
            f.readline()
            for line in f:
                if line.strip():
                    u, v = map(int, line.split())
                    nodes.update([u, v])

        comm_nodes = set()
        counter = 0
        with open(self.comm_file, "r") as f:
            for line in f:
                if counter == self.community_size:
                    break
                counter += 1
                if line.strip():
                    comm_nodes.update(map(int, line.split()))
        
        nodes = sorted(nodes.intersection(comm_nodes))
        next_id = 1
        for old_id in nodes:
            self.mapping[old_id] = next_id
            next_id += 1

    def remap_edge_file(self):
        adj = defaultdict(set)
        max_node = num_edges = 0

        with open(self.edge_file, "r") as f:
            f.readline()
            for line in f:
                if not line.strip():
                    continue
                try:
                    u, v = map(int, line.split())
                    if u in self.mapping and v in self.mapping:
                        u2, v2 = self.mapping[u], self.mapping[v]
                        adj[u2].add(v2)
                        adj[v2].add(u2)
                        max_node = max(max_node, u2, v2)
                        num_edges += 1
                except KeyError as e:
                    pass

        with open(self.new_edge_file, "w") as f:
            f.write(f"{len(self.mapping)} {num_edges}")
            for i in range(max_node+1):
                neighbors = sorted(adj[i] if i in adj else [])
                f.write(" ".join(map(str, neighbors)) + "\n")

    def remap_comm_file(self):
        counter = 0
        with open(self.comm_file, "r") as fin, open(self.new_comm_file, "w") as fout:
            for line in fin:
                if counter == self.community_size:
                    break
                counter += 1
                if not line.strip():
                    continue
                try:
                    new_line = " ".join(str(self.mapping[int(x)]) for x in line.split())
                    fout.write(new_line+"\n")
                except KeyError as e:
                    pass

    def process(self):
        self.build_mapping()
        self.remap_edge_file()
        self.remap_comm_file()

class Node:
    neighbors: list[int]
    communities: list[int]
    degree: int
    stubs: list[list[int]]

    def __init__(self):
        self.neighbors = []
        self.communities = []
        self.degree = 0
        self.stubs = []

    def add_community(self, comm_id):
        self.communities.append(comm_id)

    def add_neighbors(self, neighbors):
        if not isinstance(neighbors, list):
            neighbors = [neighbors,]
        neighbors = [int(neighbor) for neighbor in neighbors if neighbor]
        self.neighbors.extend(neighbors)
        self.degree += len(neighbors)

class Graph:
    nodes: list[Node]
    communities: dict[int, set[int]]
    adjacency_matrix: np.ndarray

    def __init__(self):
        self.nodes = []
        self.communities = {}
        self.adjacency_matrix = np.array([])

    def get_node(self, node_id):
        return self.nodes[node_id-1]

    def add_node(self):
        node = Node()
        self.nodes.append(node)
        return node

    def add_community(self, comm_id, present_nodes):
        present_nodes = {int(x) for x in present_nodes}
        self.communities.update({comm_id: present_nodes})

    def update_adjacency_matrix(self):
        n = len(self.nodes)
        mat = np.zeros((n, n), dtype=float)
        for node1_idx, node in enumerate(self.nodes, 0):
            for neighbor_id in node.neighbors:
                mat[node1_idx, neighbor_id-1] += 1
        self.adjacency_matrix = mat

    def generate_stub_information(self):
        for node in self.nodes:
            for neighbor_id in node.neighbors:
                neighbor = self.get_node(neighbor_id)
                node.stubs.append(list(neighbor.communities))

    def visualize_graph(self, edges_to_be_added = [], ax = None):
        G = nx.Graph()
        num_comm = len(self.communities)
        for node_id in range(len(self.nodes)):
            G.add_node(node_id+1)
        if edges_to_be_added:
            for (n1, n2) in edges_to_be_added:
                G.add_edge(n1, n2)
        else:
            for node_id, node in enumerate(self.nodes, 1):
                for neighbor_id in node.neighbors:
                    G.add_edge(node_id, neighbor_id)

        L = nx.Graph()
        L.add_nodes_from(G.nodes())
        for u, v in G.edges():
            L.add_edge(u, v, weight=1.0)

        nonempty_comms = []
        for comm_id in range(1, num_comm+1):
            members = [n_id for n_id, node in enumerate(self.nodes, 1) if comm_id in node.communities]
            if members:
                nonempty_comms.append((comm_id, members))

        if not nonempty_comms:
            pos = nx.spring_layout(G, seed=42)
        else:
            center_pos = {}
            center_nodes = []
            radius = max(1.5, np.sqrt(len(self.nodes))*1.0)
            for i, (comm_id, members) in enumerate(nonempty_comms):
                cname = f"COMM_CENTER_{comm_id}"
                center_nodes.append(cname)
                L.add_node(cname)
                for m in members:
                    L.add_edge(cname, m, weight=5.0)
                angle = 2*np.pi*(i/len(nonempty_comms))
                center_pos[cname] = (radius*np.cos(angle), radius*np.sin(angle))

            try:
                pos_all = nx.spring_layout(L, pos=center_pos, fixed=list(center_pos.keys()), weight='weight', seed=42, iterations=200)
                pos = {n: pos_all[n] for n in G.nodes()}
            except Exception:
                print("Using default layout")
                pos = nx.spring_layout(G, seed=42)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        cmap = plt.cm.get_cmap('tab20')
        community_colors = [cmap(i/max(1, num_comm)) for i in range(num_comm)]

        for comm_id in range(1, num_comm+1):
            members = [n_id for n_id, node in enumerate(self.nodes, 1) if comm_id in node.communities]
            if not members:
                continue

            points = np.array([pos[m] for m in members])
            center = points.mean(axis=0)

            if len(members) == 1:
                width = height = 0.4
                angle = 0.0
            elif len(members) == 2:
                p0, p1 = points
                d = np.linalg.norm(p1-p0)
                width = max(0.4, d*1.6)
                height = max(0.4*0.6, d*0.6)
                angle = np.degrees(np.arctan2(p1[1]-p0[1], p1[0]-p0[0]))
            else:
                cov = np.cov(points, rowvar=False)
                cov += np.eye(2)*epsilon
                eigvals, eigvecs = np.linalg.eigh(cov)
                order = eigvals.argsort()[::-1]
                eigvals = eigvals[order]
                eigvecs = eigvecs[:, order]
                angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
                width = 2.0*np.sqrt(max(eigvals[0], 0.0))*2.5+0.4
                height = 2.0*np.sqrt(max(eigvals[1], 0.0))*2.5+0.4

            ell = Ellipse(xy=center, width=width, height=height, angle=angle, facecolor=community_colors[comm_id-1], alpha=0.25, edgecolor='none', zorder=0)
            ax.add_patch(ell)

        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', width=0.5, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightgray', node_size=500, edgecolors='black')
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_color='black')

        legend_elements = [
            Patch(facecolor=community_colors[i], alpha=0.25, label=f'Community {i+1}')
            for i in range(num_comm)
            if any((i+1) in getattr(node, "communities", []) for node in self.nodes)
        ]
        if legend_elements:
            ax.legend(handles=legend_elements, loc='best')

        ax.set_title(f"Graph with {len(self.nodes)} Nodes, {G.number_of_edges()} Edges, and {num_comm} Communities")
        ax.set_axis_off()

class GraphReader:
    community_size: int
    child_directory: str

    def __init__(self, community_size=10, child_directory="dblp"):
        self.community_size = community_size
        self.child_directory = child_directory

    def read_metis_file(self):
        graph = Graph()

        with open(f"dataset/{self.child_directory}/small_graphs/graph{self.community_size}.metis", "r") as f:
            [num_nodes, num_edges] = f.readline().split("\n")[0].split(" ")
            print(f"Generating graph with {num_nodes} nodes and {num_edges} edges.")
            lines = f.readlines()
            for line in lines:
                line = line.split("\n")[0]
                node = graph.add_node()
                node.add_neighbors(line.split(" "))
        graph.update_adjacency_matrix()

        with open(f"dataset/{self.child_directory}/small_graphs/communities{self.community_size}.txt", "r") as f:
            lines = f.readlines()
            comm_id = 0
            for line in lines:
                comm_id += 1
                present_nodes = line.split("\n")[0].split(" ")
                graph.add_community(comm_id, present_nodes)
                for node_id in present_nodes:
                    node = graph.get_node(int(node_id))
                    node.add_community(comm_id)
            print(f"Updated {comm_id} communities.")

        graph.generate_stub_information()

        return graph

    def write_to_metis(self, edgelist):
        adj = defaultdict(set)
        for u, v in edgelist:
            if u == v:
                continue
            adj[u].add(v)
            adj[v].add(u)
        n = max(adj.keys())
        m = sum(len(neigh) for neigh in adj.values()) // 2
        with open(f"dataset/{self.child_directory}/small_graphs/predicted{self.community_size}.metis", "w") as f:
            f.write(f"{n} {m}\n")
            for i in range(1, n+1):
                neighbors = sorted(adj[i])
                f.write(" ".join(map(str, neighbors)) + "\n")

def stub_matching(graph):
    n = len(graph.nodes)
    randomly_ordered_nodes = list(range(1, n+1))
    random.shuffle(randomly_ordered_nodes)
    predicted_edges = []
    predicted_matrix = np.zeros((n, n), dtype=int)
    for node_id in randomly_ordered_nodes:
        used_nodes = {node_id}  # dont allow multi edges
        node = graph.get_node(node_id)
        expected_communities = node.communities
        for stub in node.stubs:
            probable_nodes = set.intersection(*(graph.communities[comm] for comm in stub))
            probable_nodes -= used_nodes
            probable_nodes = [v for v in probable_nodes if graph.get_node(v).communities == stub]
            chosen_node_id = None
            best_counter = 0
            for dest_node_id in probable_nodes:
                dest_node = graph.get_node(dest_node_id)
                counter = dest_node.stubs.count(expected_communities)
                if counter > best_counter:
                    chosen_node_id = dest_node_id
                    best_counter = counter
            if chosen_node_id is None:
                pdb.set_trace()
            chosen_node = graph.get_node(chosen_node_id)
            chosen_node.stubs.remove(expected_communities)
            used_nodes.add(chosen_node_id)
            predicted_edges.append((node_id, chosen_node_id))
            predicted_matrix[node_id-1, chosen_node_id-1] = 1
            predicted_matrix[chosen_node_id-1, node_id-1] = 1
        node.stubs = []
    return predicted_edges, predicted_matrix

def plot_spectral_density(adjacency_matrix, ax=None):
    hamiltonian_matrix = sp.csr_matrix(adjacency_matrix)
    dos = kwant.kpm.SpectralDensity(hamiltonian_matrix)
    k = min(50, adjacency_matrix.shape[0]-2)
    eigenvalues = eigsh(hamiltonian_matrix, k=k, return_eigenvectors=False)
    padding = 0.1 * (eigenvalues.max() - eigenvalues.min())
    energy_range = (eigenvalues.min() - padding, eigenvalues.max() + padding)
    energies = np.linspace(energy_range[0], energy_range[1], 200)
    rho = dos(energies)
    if ax is None:
        ax = plt.gca()
    ax.plot(energies, rho)
    ax.set_xlabel("Eigenvalues")
    ax.set_ylabel("Density")
    ax.set_title("Eigenvalue Distribution of Graph (KPM)")

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
        predicted_edges, predicted_matrix = stub_matching(graph)
        end_time = time.time()

        num_nodes = len(graph.nodes)
        original_edges = set()
        for i in range(num_nodes):
            for j in range(i, num_nodes):
                if graph.adjacency_matrix[i, j] == 1:
                    original_edges.add((i+1, j+1))
        predicted_edges = set([(min(a, b), max(a, b)) for a, b in predicted_edges])
        print(f"Time taken: {end_time - start_time} seconds. Newly seen edges: {(len(predicted_edges - original_edges)*100.0)/len(predicted_edges)}%")

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        plot_spectral_density(graph.adjacency_matrix, ax=axes[0])
        plot_spectral_density(predicted_matrix, ax=axes[1])
        ymin = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
        ymax = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
        axes[0].set_ylim(ymin, ymax)
        axes[1].set_ylim(ymin, ymax)
        plt.tight_layout()
        plt.show()

