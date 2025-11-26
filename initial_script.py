import argparse
import copy
import kwant
import math
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
percent_eigenvalue_plot = 10

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
    predicted_neighbors: list[int]
    communities: list[int]
    degree: int
    static_stubs: list[list[int]]
    stubs: list[list[int]]
    connectivity: set[int]
    static_order: dict[int, int]
    order: dict[int, int]

    def __init__(self):
        self.neighbors = []
        self.predicted_neighbors = []
        self.communities = []
        self.degree = 0
        self.static_stubs = []
        self.stubs = []
        self.connectivity = set()
        self.static_order = {}
        self.order = {}

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
                node.static_stubs.append(list(neighbor.communities))
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

def increase_node_order(node, overlap_size):
    if node.order.get(overlap_size, 0) == 0:
        node.order.update({overlap_size: 0})
    node.order[overlap_size] += 1

def update_unmatched_stubs(unmatched_stubs, b_id, overlap_size):
    for i, (v_id, v_size, v_num) in enumerate(unmatched_stubs):
        if v_id == b_id and v_size == overlap_size:
            unmatched_stubs[i] = (v_id, v_size, v_num+1)
            return
    unmatched_stubs.append((b_id, overlap_size, 1))

def break_prior_edges(unmatched_stubs, possible_nodes, overlap_size, src_comm_len, predicted_edges, predicted_matrix, req_num_dest_nodes):
    counter = 0
    chosen_nodes = []
    while counter < req_num_dest_nodes:
        c_id = possible_nodes.pop(0)
        chosen_node = graph.get_node(c_id)
        # this condition could be something to look at: set(graph.get_node(v).communities) != set(node.communities)
        broken_possibilites = [v for v in chosen_node.predicted_neighbors if len(graph.get_node(v).communities) == src_comm_len]
        if len(broken_possibilites) == 0:
            continue
        b_id = random.choice(broken_possibilites)
        if b_id in possible_nodes:
            possible_nodes.remove(b_id)
            counter += 1
        broken_node = graph.get_node(b_id)
        broken_node.predicted_neighbors.remove(c_id)
        chosen_node.predicted_neighbors.remove(b_id)
        predicted_edges.remove((min(b_id, c_id), max(b_id, c_id)))
        predicted_matrix[c_id-1, b_id-1] = 0
        predicted_matrix[b_id-1, c_id-1] = 0
        update_unmatched_stubs(unmatched_stubs, b_id, overlap_size)
        increase_node_order(broken_node, overlap_size)
        increase_node_order(chosen_node, src_comm_len)
        counter += 1
        chosen_nodes.append(c_id)
    return chosen_nodes

def reduce_unmatched_stubs(unmatched_stubs, open_stubs, src_comm_len):
    for i in range(len(unmatched_stubs)-1, -1, -1):
        v_id, v_size, v_num = unmatched_stubs[i]
        if v_id in open_stubs and v_size == src_comm_len:
            v_num -= 1
            if v_num > 0:
                unmatched_stubs[i] = (v_id, v_size, v_num)
            else:
                unmatched_stubs.pop(i)

def has_unmatched_entry(unmatched_stubs, u_id, src_comm_len):
    for i, (v_id, v_size, v_num) in enumerate(unmatched_stubs):
        if u_id == v_id and v_size == src_comm_len:
            return True
    return False

def find_unmatched_stubs(graph):
    unmatched_stubs = []
    for node_id, node in enumerate(graph.nodes, 1):
        for overlap_size, num_items in node.order.items():
            unmatched_stubs.append((node_id, overlap_size, num_items))
    return unmatched_stubs

def find_involved_nodes(graph, node, node_id):
    all_nodes = set.union(*(graph.communities[comm] for comm in node.connectivity)) - {node_id}
    return set([v for v in all_nodes if set(graph.get_node(v).communities).issubset(node.connectivity)])

def update_node_order(node, overlap_size, num_connections):
    node.order[overlap_size] -= num_connections
    if node.order[overlap_size] == 0:
        node.order.pop(overlap_size)

def connect_to_chosen_nodes(chosen_nodes, node, node_id, src_comm_len, predicted_edges, predicted_matrix):
    for c_id in chosen_nodes:
        chosen_node = graph.get_node(c_id)
        chosen_node.predicted_neighbors.append(node_id)
        node.predicted_neighbors.append(c_id)
        update_node_order(chosen_node, src_comm_len, 1)
        predicted_edges.append((min(node_id, c_id), max(node_id, c_id)))
        predicted_matrix[node_id-1, c_id-1] = 1
        predicted_matrix[c_id-1, node_id-1] = 1

def update_stub_info(graph):
    for node in graph.nodes:
        for stub in node.static_stubs:
            stub_len = len(stub)
            if stub_len not in node.order:
                node.static_order[stub_len] = 0
                node.order[stub_len] = 0
            node.static_order[stub_len] += 1
            node.order[stub_len] += 1
            node.connectivity.update(stub)

def look_for_open_stubs(union_nodes, unmatched_stubs, node, src_comm_len, overlap_size):
    possible_nodes = []
    open_stubs = []
    for u_id in union_nodes:
        u = graph.get_node(u_id)
        if set(u.communities).issubset(node.connectivity) and u.static_order.get(src_comm_len, 0) > 0 and len(u.communities) == overlap_size:
            if has_unmatched_entry(unmatched_stubs, u_id, src_comm_len):
                open_stubs.append(u_id)
            else:
                possible_nodes.append(u_id)
    return open_stubs, possible_nodes

def stub_matching2(graph):
    update_stub_info(graph)
    # Phase 1: Create first set of connections without breaking any edges
    n = len(graph.nodes)
    num_comm = len(graph.communities)
    randomly_ordered_nodes = list(range(1, n+1))
    random.shuffle(randomly_ordered_nodes)
    predicted_edges = []
    predicted_matrix = np.zeros((n, n), dtype=int)
    for node_id in randomly_ordered_nodes:
        node = graph.get_node(node_id)
        src_comm_len = len(node.communities)
        involved_nodes = find_involved_nodes(graph, node, node_id)
        for overlap_size in sorted(node.order, reverse=True):
            involved_nodes -= set(node.predicted_neighbors)
            req_num_dest_nodes = int(node.order[overlap_size])
            possible_nodes = [v for v in involved_nodes if len(graph.get_node(v).communities) == overlap_size and graph.get_node(v).order.get(src_comm_len, 0) > 0]
            num_choices = min(len(possible_nodes), req_num_dest_nodes)
            if num_choices == 0:
                continue
            chosen_nodes = list(map(int, np.random.choice(possible_nodes, size=num_choices, replace=False)))
            connect_to_chosen_nodes(chosen_nodes, node, node_id, src_comm_len, predicted_edges, predicted_matrix)
            update_node_order(node, overlap_size, num_choices)

    unmatched_stubs = find_unmatched_stubs(graph)
    # Phase 2: Break prior edges to fix subgraph
    if len(unmatched_stubs) > 0:
        print("Going through phase 2...")
        print(f"Percent of unmatched stubs: {200.0*sum([v[2] for v in unmatched_stubs])/sum(sum(graph.adjacency_matrix))}")
        temp_graph = copy.deepcopy(graph)
        temp_unmatched_stubs = copy.deepcopy(unmatched_stubs)
        temp_predicted_edges = copy.deepcopy(predicted_edges)
        temp_predicted_matrix = copy.deepcopy(predicted_matrix)
        time1 = time.time()
        print(f"Before Random Remaining: {sum([v[2] for v in unmatched_stubs])}")
        random_matching(temp_graph, temp_unmatched_stubs, temp_predicted_edges, temp_predicted_matrix)
        time2 = time.time()
        initial_open_stubs = sum([v[2] for v in unmatched_stubs])
        careful_matching(graph, unmatched_stubs, predicted_edges, predicted_matrix, initial_open_stubs)
        random_matching(graph, unmatched_stubs, predicted_edges, predicted_matrix)
        time3 = time.time()
        print(f"Time : Random matching is {(time3-time2)/(time2-time1)} times faster than Careful matching.")

    return temp_predicted_edges, temp_predicted_matrix, predicted_edges, predicted_matrix

def careful_matching(graph, unmatched_stubs, predicted_edges, predicted_matrix, initial_open_stubs):
    while unmatched_stubs:
        if 1.0*sum([v[2] for v in unmatched_stubs])/initial_open_stubs < 0.1:
            return
        node_id, overlap_size, req_num_dest_nodes = unmatched_stubs.pop(0)
        node = graph.get_node(node_id)
        src_comm_len = len(node.communities)
        union_nodes = set.union(*(graph.communities[comm] for comm in node.connectivity)) - set(node.predicted_neighbors) - {node_id}
        open_stubs, possible_nodes = look_for_open_stubs(union_nodes, unmatched_stubs, node, src_comm_len, overlap_size)
        if len(open_stubs) > 0:
            open_stubs = open_stubs[:min(req_num_dest_nodes, len(open_stubs))]
            connect_to_chosen_nodes(open_stubs, node, node_id, src_comm_len, predicted_edges, predicted_matrix)
            reduce_unmatched_stubs(unmatched_stubs, open_stubs, src_comm_len)
            update_node_order(node, overlap_size, len(open_stubs))
            req_num_dest_nodes -= len(open_stubs)
        if req_num_dest_nodes == 0:
            continue
        # If not enough open stubs, break some pre-existing edges
        random.shuffle(possible_nodes)
#        chosen_nodes = list(map(int, np.random.choice(possible_nodes, size=req_num_dest_nodes, replace=False)))
        chosen_nodes = break_prior_edges(unmatched_stubs, possible_nodes, overlap_size, src_comm_len, predicted_edges, predicted_matrix, req_num_dest_nodes)
        connect_to_chosen_nodes(chosen_nodes, node, node_id, src_comm_len, predicted_edges, predicted_matrix)
        update_node_order(node, overlap_size, req_num_dest_nodes)

def random_matching(graph, unmatched_stubs, predicted_edges, predicted_matrix):
    # Phase 2: Randomly connect rest of the stubs
    while unmatched_stubs:
        unmatched_stubs.sort(key=lambda x: x[2], reverse=True)
        src_node_id, _, src_stubs = unmatched_stubs.pop(0)
        src_node = graph.get_node(src_node_id)
        # Break some uniform nodes to make sure there are no multi/self edges
        if src_stubs > len(unmatched_stubs):
            unmatched_stubs.extend(break_random_edges(graph, src_node, src_node_id, src_stubs, [v[0] for v in unmatched_stubs], predicted_edges, predicted_matrix))
        idx = 0
        while idx < src_stubs:
            dest_node_id, dest_overlap_size, dest_stubs = unmatched_stubs[idx]
            dest_node = graph.get_node(dest_node_id)
            dest_node.predicted_neighbors.append(src_node_id)
            src_node.predicted_neighbors.append(dest_node_id)
            predicted_edges.append((min(src_node_id, dest_node_id), max(src_node_id, dest_node_id)))
            predicted_matrix[src_node_id-1, dest_node_id-1] = 1
            predicted_matrix[dest_node_id-1, src_node_id-1] = 1
            if dest_stubs == 1:
                unmatched_stubs.pop(idx)
                src_stubs -= 1
            else:
                unmatched_stubs[idx] = (dest_node_id, dest_overlap_size, dest_stubs-1)
                idx += 1

def break_random_edges(graph, src_node, src_node_id, src_stubs, pre_existing_stubs, predicted_edges, predicted_matrix):
    newly_open_stubs = []
    remaining_stubs = math.ceil((src_stubs-len(pre_existing_stubs))/2)
    possible_nodes = list(set(range(1, len(graph.nodes)+1)) - set(src_node.predicted_neighbors) - set(pre_existing_stubs) - {src_node_id})
    chosen_nodes = list(map(int, np.random.choice(possible_nodes, size=remaining_stubs, replace=False)))
    for c_id in chosen_nodes:
        newly_open_stubs.append((c_id, 0, 1))
        chosen_node = graph.get_node(c_id)
        b_id = random.choice(chosen_node.predicted_neighbors)
        newly_open_stubs.append((b_id, 0, 1))
        broken_node = graph.get_node(b_id)
        broken_node.predicted_neighbors.remove(c_id)
        chosen_node.predicted_neighbors.remove(b_id)
        predicted_edges.remove((min(b_id, c_id), max(b_id, c_id)))
        predicted_matrix[b_id-1, c_id-1] = 0
        predicted_matrix[c_id-1, b_id-1] = 0
    return newly_open_stubs

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

