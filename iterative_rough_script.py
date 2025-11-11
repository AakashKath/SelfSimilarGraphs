import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pdb
import random
from matplotlib.patches import Patch, Ellipse

eta = 0.1
delta = 0.1
l2_lambda = 0.1
entropy_lambda = 0.1
epsilon = 1e-8

class Graph:
    nodes: list[int]
    communities: dict[int, list[int]]
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
        present_nodes = [int(x) for x in present_nodes]
        self.communities.update({comm_id: present_nodes})

    def update_adjacency_matrix(self):
        n = len(self.nodes)
        mat = np.zeros((n, n), dtype=float)
        for node1_idx, node in enumerate(self.nodes, 0):
            for neighbor_id in node.neighbors:
                mat[node1_idx, neighbor_id-1] += 1
        self.adjacency_matrix = mat

class Node:
    neighbors: list[int]
    communities: list[int]
    degree: int

    def __init__(self):
        self.neighbors = []
        self.communities = []
        self.degree = 0

    def add_community(self, comm_id):
        self.communities.append(comm_id)

    def add_neighbors(self, neighbors):
        if not isinstance(neighbors, list):
            neighbors = [neighbors,]
        neighbors = [int(neighbor) for neighbor in neighbors if neighbor]
        self.neighbors.extend(neighbors)
        self.degree += len(neighbors)

def read_metis_file(graph):
    #with open("anon_graph_inner_digitalhumanities_20211018_undirected.metis" , "r") as f:
    with open("graph7.metis", "r") as f:
        [num_nodes, num_edges] = f.readline().split("\n")[0].split(" ")
        print(f"Generating graph with {num_nodes} nodes and {num_edges} edges.")
        lines = f.readlines()
        for line in lines:
            line = line.split("\n")[0]
            node = graph.add_node()
            node.add_neighbors(line.split(" "))
    graph.update_adjacency_matrix()

    #with open("anon_graph_inner_digitalhumanities_20211018_undirected.metis-communities.nl", "r") as f:
    with open("communities7.txt", "r") as f:
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
    return comm_id

def visualize_graph(graph, num_comm, edges_to_be_added = [], ax = None):
    G = nx.Graph()
    for node_id in range(len(graph.nodes)):
        G.add_node(node_id+1)
    if edges_to_be_added:
        for (n1, n2) in edges_to_be_added:
            G.add_edge(n1, n2)
    else:
        for node_id, node in enumerate(graph.nodes, 1):
            for neighbor_id in node.neighbors:
                G.add_edge(node_id, neighbor_id)

    L = nx.Graph()
    L.add_nodes_from(G.nodes())
    for u, v in G.edges():
        L.add_edge(u, v, weight=1.0)

    nonempty_comms = []
    for comm_id in range(1, num_comm+1):
        members = [n_id for n_id, node in enumerate(graph.nodes, 1) if comm_id in node.communities]
        if members:
            nonempty_comms.append((comm_id, members))

    if not nonempty_comms:
        pos = nx.spring_layout(G, seed=42)
    else:
        center_pos = {}
        center_nodes = []
        radius = max(1.5, np.sqrt(len(graph.nodes))*1.0)
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
        members = [n_id for n_id, node in enumerate(graph.nodes, 1) if comm_id in node.communities]
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
        if any((i+1) in getattr(node, "communities", []) for node in graph.nodes)
    ]
    if legend_elements:
        ax.legend(handles=legend_elements, loc='best')

    ax.set_title(f"Graph with {len(graph.nodes)} Nodes, {G.number_of_edges()} Edges, and {num_comm} Communities")
    ax.set_axis_off()

def get_community_weights(graph):
    comm_weights = {}
    for node in graph.nodes:
        if len(node.communities) == 0:
            continue
        contribution_by_node = node.degree/(2*len(node.communities))
        for comm in node.communities:
            current_weight = comm_weights.get(comm, 0)
            comm_weights.update({comm: current_weight + contribution_by_node})
    return comm_weights

# TODO: Not efficient at all, plz fix me when you have time
def compute_pairwise_matrix(graph, num_comm, edges_to_be_added = None):
    mat = np.zeros((num_comm, num_comm), dtype=float)
    total_edges = within = between = 0
    if edges_to_be_added:
        for (node1_id, node2_id) in edges_to_be_added:
            node1 = graph.get_node(node1_id)
            node2 = graph.get_node(node2_id)
            for comm_i in node1.communities:
                for comm_j in node2.communities:
                    if comm_i == comm_j:
                        within += 2
                    else:
                        between += 2
                    mat[comm_i-1, comm_j-1] += 1.0/(len(node1.communities)*len(node2.communities))
                    mat[comm_j-1, comm_i-1] += 1.0/(len(node1.communities)*len(node2.communities))
    else:
        original_edges = []
        for idx, node in enumerate(graph.nodes, 1):
            total_edges += len(node.neighbors)
            for neighbor in node.neighbors:
                if neighbor > idx:
                    original_edges.append((idx, neighbor))
                neighbor_node = graph.get_node(neighbor)
                for comm_i in node.communities:
                    for comm_j in neighbor_node.communities:
                        if comm_i == comm_j:
                            within += 1
                        else:
                            between += 1
                        mat[comm_i-1, comm_j-1] += 1.0/(len(node.communities)*len(neighbor_node.communities))
    return np.round(mat, 2), total_edges//2

def compute_community_id_weights(pairwise_matrix, membership_matrix):
    comm_ids = []
    comm_weights = []
    num_nodes = len(membership_matrix)
    for row in range(len(pairwise_matrix)):
        for col in range(len(pairwise_matrix[row])):
            comm_ids.append(f"{row+1}_{col+1}")
            total_membership = 0.0
            for i in range(num_nodes):
                for j in range(i, num_nodes):
                    node1_membership = membership_matrix[i, row]
                    node2_membership = membership_matrix[j, col]
                    total_membership += node1_membership*node2_membership
            comm_weights.append(total_membership)
    return comm_ids, comm_weights

def compute_comm_node_weights(graph, membership_matrix):
    comm_node_weights = {}
    for idx, node in enumerate(graph.nodes, 1):
        for comm in node.communities:
            if comm not in comm_node_weights:
                comm_node_weights[comm] = [[], []]
            comm_node_weights[comm][0].append(idx)
            comm_node_weights[comm][1].append(round(len(node.neighbors)*membership_matrix[idx-1, comm-1], 2))
    return comm_node_weights

def pick_nodes(comm_node_weights, comm1, comm2):
    nodes1, weights1 = list(comm_node_weights[comm1][0]), list(comm_node_weights[comm1][1])
    nodes2, weights2 = list(comm_node_weights[comm2][0]), list(comm_node_weights[comm2][1])
    node1 = random.choices(nodes1, weights=weights1, k=1)[0]

    if node1 in nodes2:
        idx = nodes2.index(node1)
        nodes2.pop(idx)
        weights2.pop(idx)
    
    if not nodes2:
        raise ValueError(f"Found empty list: {comm2}, nodes2: {nodes2}, weights2: {weights2}")
    node2 = random.choices(nodes2, weights=weights2, k=1)[0]

    return (node1, node2)

def node_pick_strat2(graph, pairwise_matrix, membership_matrix):
    observed_degrees = np.array([node.degree for node in graph.nodes])
    degree_weighted = membership_matrix*observed_degrees[:, np.newaxis]
    total_mass = np.sum(degree_weighted)
    scale_factor = np.sum(membership_matrix)/total_mass
    degree_weighted *= scale_factor

    num_nodes = len(degree_weighted)
    W = degree_weighted @ pairwise_matrix @ degree_weighted.T
    np.fill_diagonal(W, 0)
    flat_weights = W.flatten()
    indices = list(range(num_nodes * num_nodes))
    chosen_idx = random.choices(indices, weights=flat_weights, k=1)[0]
    return ((chosen_idx//num_nodes)+1, (chosen_idx%num_nodes)+1)

def node_pick_strat3(graph, needed_edges, pairwise_matrix, membership_matrix):
    edges = []
    num_comm = len(pairwise_matrix)
    flat_weights = pairwise_matrix.flatten()
    indices = list(range(num_comm*num_comm))
    random_community_pairs = random.choices(indices, weights=flat_weights, k=needed_edges)
    for comm_pair in random_community_pairs:
        comm1 = graph.communities[(comm_pair//num_comm)+1]
        comm2 = graph.communities[(comm_pair%num_comm)+1]
        while True:
            node1 = random.choice(comm1)
            node2 = random.choice(comm2)
            if node1 != node2:
                edges.append((node1, node2))
                break
    #counter = {}
    #for comm_pair in random_community_pairs:
    #    comm1 = (comm_pair//num_comm)+1
    #    comm2 = (comm_pair%num_comm)+1
    #    key1 = min(comm1, comm2)
    #    key2 = max(comm1, comm2)
    #    if f"{key1}_{key2}" not in counter:
    #        counter[f"{key1}_{key2}"] = 0
    #    counter[f"{key1}_{key2}"] += 1
    #print(counter)
    #pdb.set_trace()
    return edges

def compute_membership_matrix(graph, num_comm):
    membership = np.zeros((len(graph.nodes), num_comm), dtype=float)
    for node_id, node in enumerate(graph.nodes, 0):
        for comm_id in node.communities:
            membership[node_id, comm_id-1] += 1.0/len(node.communities)
    return membership

def compute_loss_matrix(adjacency_matrix, membership_matrix, pairwise_matrix):
    return adjacency_matrix - membership_matrix @ pairwise_matrix @ membership_matrix.T

def compute_updated_pairwise_matrix(adjacency_matrix, membership_matrix, pairwise_matrix):
    gradient_pairwise_matrix = 2.0*eta*(membership_matrix.T @ compute_loss_matrix(adjacency_matrix, membership_matrix, pairwise_matrix) @ membership_matrix)
#    gradient_pairwise_matrix -= 2.0*eta*l2_lambda*pairwise_matrix
#    gradient_pairwise_matrix -= (entropy_lambda*(1+np.log(pairwise_matrix+epsilon)))
    gradient_pairwise_matrix = gradient_pairwise_matrix/(np.linalg.norm(gradient_pairwise_matrix)+epsilon)
#    change_in_pairwise_matrix = (np.abs(gradient_pairwise_matrix) > delta).any()
    change_in_pairwise_matrix = True
    if change_in_pairwise_matrix:
        pairwise_matrix += gradient_pairwise_matrix
        # Make C symmetric to keep the undirected aspect
        pairwise_matrix = np.maximum(pairwise_matrix, 0)
        pairwise_matrix = (pairwise_matrix+pairwise_matrix.T)/2
    return pairwise_matrix, change_in_pairwise_matrix

def project_membership_matrix(original_membership_matrix, updated_membership_matrix):
    projected_matrix = updated_membership_matrix.astype(float).copy()
    for idx, (original_row, updated_row) in enumerate(zip(original_membership_matrix, updated_membership_matrix)):
        mask = original_row != 0
        if not np.any(mask):
            continue
        allowed_values = updated_row[mask]
        sorted_vals = np.sort(allowed_values)[::-1]
        cs_vals = np.cumsum(sorted_vals)
        rho = np.where(sorted_vals > (cs_vals - 1) / (np.arange(len(sorted_vals)) + 1))[0]
        if len(rho) > 0:
            rho = rho[-1]
            theta = (cs_vals[rho] - 1) / (rho + 1.0)
            projected_subvector = np.maximum(allowed_values - theta, 0)
        else:
            # Fallback: uniform distribution if projection fails
            projected_subvector = np.ones_like(allowed_values) / len(allowed_values)
        projected_matrix[idx, mask] = projected_subvector
        projected_matrix[idx, ~mask] = 0
        row_sum = projected_matrix[idx, mask].sum()
        if row_sum > 0:
            projected_matrix[idx, mask] /= row_sum
        else:
            # Fallback: if all zeros after projection, use uniform distribution
            projected_matrix[idx, mask] = 1.0 / len(projected_subvector)
    return projected_matrix

def compute_updated_membership_matrix(original_membership_matrix, adjacency_matrix, membership_matrix, pairwise_matrix):
    gradient_membership_matrix = 4.0*eta*(compute_loss_matrix(adjacency_matrix, membership_matrix, pairwise_matrix) @ membership_matrix @ pairwise_matrix)
#    gradient_membership_matrix -= 2.0*eta*l2_lambda*membership_matrix
#    gradient_membership_matrix -= (entropy_lambda*(1+np.log(membership_matrix+epsilon)))
    gradient_membership_matrix = gradient_membership_matrix/(np.linalg.norm(gradient_membership_matrix)+epsilon)
#    change_in_membership_matrix = (np.abs(gradient_membership_matrix) > delta).any()
    change_in_membership_matrix = True
    if change_in_membership_matrix:
        membership_matrix += gradient_membership_matrix
        # normalzie membership_matrix
        membership_matrix = project_membership_matrix(original_membership_matrix, membership_matrix)
    return membership_matrix, change_in_membership_matrix

def plot_loss_history(loss_history):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, "b-", linewidth=2)
    plt.ylabel("Loss")
    plt.xlabel("Iteration")
    plt.title("Loss Convergence")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    print(f"Loss is a funny thing: {max(loss_history)}, {min(loss_history)}")

def iterative_update_membership_pairwise_matrix(adjacency_matrix, membership_matrix, pairwise_matrix):
    updated_membership_matrix = membership_matrix.copy()
    updated_pairwise_matrix = pairwise_matrix.copy()
    change_in_membership_matrix = change_in_pairwise_matrix = True
    iteration_count = 0
    previous_loss = np.inf

    loss_history = []
    while iteration_count < 1000 and (change_in_pairwise_matrix or change_in_membership_matrix):
#        updated_membership_matrix, change_in_membership_matrix = compute_updated_membership_matrix(membership_matrix.copy(), adjacency_matrix.copy(), updated_membership_matrix.copy(), updated_pairwise_matrix.copy())
        updated_pairwise_matrix, change_in_pairwise_matrix = compute_updated_pairwise_matrix(adjacency_matrix.copy(), updated_membership_matrix.copy(), updated_pairwise_matrix.copy())
        current_loss = np.linalg.norm(compute_loss_matrix(adjacency_matrix, updated_membership_matrix, updated_pairwise_matrix))
        loss_history.append(current_loss)
#        if current_loss > previous_loss and iteration_count > 10:
#            global eta
#            eta *= 0.9
#            print(f"Iteration {iteration_count}: Loss increased to {current_loss:.4f}, reducing eta to {eta:.6f}")
        previous_loss = current_loss
        iteration_count += 1
    print(f"Final iteration: {iteration_count}")
    plot_loss_history(loss_history)
    return np.round(updated_membership_matrix, 2), np.round(updated_pairwise_matrix, 2)

def test_edge_weight_metric(graph, membership_matrix, generated_edges=[]):
    num_comm = len(graph.communities)
    avg_original_matrix = np.zeros((num_comm, num_comm), dtype=[("weight", float), ("count", int)])
    for comm1_id, comm1 in graph.communities.items():
        for comm2_id, comm2 in graph.communities.items():
            total_weight = 0.0
            encountered_edges = 0
            for node1 in comm1:
                for node2 in comm2:
                    if graph.adjacency_matrix[node1-1, node2-1]:
                        total_weight += 1.0/(membership_matrix[node1-1, comm1_id-1]*membership_matrix[node2-1, comm2_id-1])
                        encountered_edges += 1
            avg_original_matrix[comm1_id-1, comm2_id-1] = (total_weight/encountered_edges, encountered_edges)
    avg_predicted_matrix = np.zeros((num_comm, num_comm), dtype=[("weight", float), ("count", int)])
    for (node1_id, node2_id) in generated_edges:
        node1 = graph.get_node(node1_id)
        node2 = graph.get_node(node2_id)
        for comm1 in node1.communities:
            for comm2 in node2.communities:
                old_value = avg_predicted_matrix[comm1-1, comm2-1]
                new_value = 1.0/(membership_matrix[node1_id-1, comm1-1]*membership_matrix[node2_id-1, comm2-1])
                avg_predicted_matrix[comm1-1, comm2-1] = ((old_value["weight"]*old_value["count"]+new_value)/(old_value["count"]+1), old_value["count"]+1)
    return avg_original_matrix["weight"], avg_predicted_matrix["weight"]

def approach1(g, num_comm):
    pairwise_matrix, total_edges = compute_pairwise_matrix(g, num_comm)
    membership_matrix = compute_membership_matrix(g, num_comm)
    #pairwise_matrix = np.eye(len(pairwise_matrix))
    pairwise_matrix = pairwise_matrix/(np.linalg.norm(pairwise_matrix)+epsilon)
    membership_matrix = membership_matrix/(np.linalg.norm(membership_matrix)+epsilon)
    updated_membership_matrix, updated_pairwise_matrix = iterative_update_membership_pairwise_matrix(g.adjacency_matrix, membership_matrix, pairwise_matrix)

#    comm_node_weights = compute_comm_node_weights(g, updated_membership_matrix)
#    comm_ids, comm_weights = compute_community_id_weights(updated_pairwise_matrix, updated_membership_matrix)

#    fig, axes = plt.subplots(1, 2, figsize=(16,8))
#    visualize_graph(g, num_comm, ax=axes[0])

    #random_community_pairs = random.choices(comm_ids, weights=comm_weights, k=total_edges)

    generated_edges = []
    needed_edges = total_edges*(1000/1000)
    while len(generated_edges) != needed_edges:
        edge = node_pick_strat2(g, updated_pairwise_matrix, updated_membership_matrix)
        generated_edges.append([min(edge), max(edge)])
    #generated_edges = node_pick_strat3(g, needed_edges, updated_pairwise_matrix, updated_membership_matrix)
    original_stubs = {}
    for node_id, node in enumerate(g.nodes, 1):
        original_stubs[node_id] = node.degree
    predicted_stubs = {}
    for edge in generated_edges:
        node1 = edge[0]
        node2 = edge[1]
        if node1 not in predicted_stubs:
            predicted_stubs[node1] = 0
        if node2 not in predicted_stubs:
            predicted_stubs[node2] = 0
        predicted_stubs[node1] += 1
        predicted_stubs[node2] += 1
    print(f"original_stubs: {dict(sorted(original_stubs.items()))}\npredicted_stubs: {dict(sorted(predicted_stubs.items()))}")

    #artificial_pairwise_matrix, _ = compute_pairwise_matrix(g, num_comm, generated_edges)
    #generated_edges: {generated_edges}
    #avg_original_matrix, avg_predicted_matrix = test_edge_weight_metric(g, membership_matrix, generated_edges)
    print(f"""
    membership_matrix:
    {membership_matrix}
    updated_memebership_matrix:
    {updated_membership_matrix}
    pairwise_matrix:
    {pairwise_matrix}
    updated_pairwise_matrix:
    {updated_pairwise_matrix}
    """)
#    print(f"""
#    comm_ids: {comm_ids}
#    comm_weights: {comm_weights}
#    comm_node_weights: {comm_node_weights}
#    adjacency_matrix: {g.adjacency_matrix}
#    membership_matrix: {membership_matrix}
#    updated_memebership_matrix: {updated_membership_matrix}
#    pairwise_matrix: {pairwise_matrix}
#    updated_pairwise_matrix: {updated_pairwise_matrix}
#    artificial_pairwise_matrix: {artificial_pairwise_matrix}
#    """)
#
#    diff = original_total = predicted_total = 0
#    for row in range(len(updated_pairwise_matrix)):
#        for col in range(len(updated_pairwise_matrix)):
#            original_total += updated_pairwise_matrix[row][col]
#            predicted_total += artificial_pairwise_matrix[row][col]
#            abs_diff = abs(updated_pairwise_matrix[row][col] - artificial_pairwise_matrix[row][col])
#            if row == col:
#                diff += abs_diff
#            else:
#                diff += abs_diff // 2
#    print(f"""
#    Original: {original_total}
#    Predicted: {predicted_total}
#    Is this acceptable error: {(diff*100.0)/total_edges}
#    """)

#    visualize_graph(g, num_comm, generated_edges, ax=axes[1])
#    plt.tight_layout()
#    plt.show()

def compute_pairwise_matrix_given_membership(adjacency_matrix, membership_matrix):
    M = membership_matrix.T @ membership_matrix
    return np.linalg.inv(M) @ (membership_matrix.T @ adjacency_matrix @ membership_matrix) @ np.linalg.inv(M)

def approach2(g, num_comm):
    membership_matrix = compute_membership_matrix(g, num_comm)
    pairwise_matrix = compute_pairwise_matrix_given_membership(g.adjacency_matrix, membership_matrix)
    pdb.set_trace()
    print("Found something")

g = Graph()
num_comm = read_metis_file(g)
approach1(g, num_comm)
#approach2(g, num_comm)

