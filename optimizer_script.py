import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pdb
import random
from matplotlib.patches import Patch, Ellipse

delta = 0.1
l2_lambda = 0.01
epsilon=1e-8
graph_size = 10

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
    with open(f"graph{graph_size}_y.metis", "r") as f:
        [num_nodes, num_edges] = f.readline().split("\n")[0].split(" ")
        print(f"Generating graph with {num_nodes} nodes and {num_edges} edges.")
        lines = f.readlines()
        for line in lines:
            line = line.split("\n")[0]
            node = graph.add_node()
            node.add_neighbors(line.split(" "))
    graph.update_adjacency_matrix()

    #with open("anon_graph_inner_digitalhumanities_20211018_undirected.metis-communities.nl", "r") as f:
    with open(f"communities{graph_size}_y.txt", "r") as f:
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

def compute_pairwise_matrix(graph, num_comm):
    mat = np.zeros((num_comm, num_comm), dtype=float)
    total_edges = 0
    original_edges = []
    for idx, node in enumerate(graph.nodes, 1):
        total_edges += len(node.neighbors)
        for neighbor in node.neighbors:
            if neighbor > idx:
                original_edges.append((idx, neighbor))
            neighbor_node = graph.get_node(neighbor)
            for comm_i in node.communities:
                for comm_j in neighbor_node.communities:
                    mat[comm_i-1, comm_j-1] += 1.0/(len(node.communities)*len(neighbor_node.communities))
    return np.round(mat, 2), total_edges//2

class Optimizer:
    def __init__(self, method="adam", eta=0.1, beta=0.9):
        self.method = method
        self.eta = eta
        self.beta = beta

        self.v_A = None
        self.v_C = None
        self.cache_A = None
        self.cache_C = None
        self.t = 0

    def initialize(self, A_shape, C_shape):
        if self.method in ['momentum', 'adam']:
            self.v_A = np.zeros(A_shape)
            self.v_C = np.zeros(C_shape)
            
        if self.method in ['adagrad', 'rmsprop', 'adam']:
            self.cache_A = np.zeros(A_shape)
            self.cache_C = np.zeros(C_shape)
            
        self.t = 0

    def update(self, g_A, g_C):
        g_A_hat = g_C_hat = None
        self.t += 1

        if self.method == 'vanilla':
            return g_A, g_C

        elif self.method == 'momentum':
            if g_A is not None:
                self.v_A = (self.beta * self.v_A + (1 - self.beta) * g_A)
            if g_C is not None:
                self.v_C = (self.beta * self.v_C + (1 - self.beta) * g_C)
            return self.v_A, self.v_C

        elif self.method == 'adagrad':
            if g_A is not None:
                self.cache_A += g_A * g_A
                g_A_hat = g_A / (np.sqrt(self.cache_A) + epsilon)
            if g_C is not None:
                self.cache_C += g_C * g_C
                g_C_hat = g_C / (np.sqrt(self.cache_C) + epsilon)
            return g_A_hat, g_C_hat

        elif self.method == 'rmsprop':
            if g_A is not None:
                self.cache_A = self.beta * self.cache_A + (1 - self.beta) * (g_A * g_A)
                cache_A_hat = self.cache_A / (1 - self.beta ** self.t)
                g_A_hat = g_A / (np.sqrt(cache_A_hat) + epsilon)
            if g_C is not None:
                self.cache_C = self.beta * self.cache_C + (1 - self.beta) * (g_C * g_C)
                cache_C_hat = self.cache_C / (1 - self.beta ** self.t)
                g_C_hat = g_C / (np.sqrt(cache_C_hat) + epsilon)
            return g_A_hat, g_C_hat

        elif self.method == 'adam':
            if g_A is not None:
                self.v_A = self.beta * self.v_A + (1 - self.beta) * g_A
                self.cache_A = self.beta * self.cache_A + (1 - self.beta) * (g_A * g_A)
                v_A_hat = self.v_A / (1 - self.beta ** self.t)
                cache_A_hat = self.cache_A / (1 - self.beta ** self.t)
                g_A_hat = v_A_hat / (np.sqrt(cache_A_hat) + epsilon)
            if g_C is not None:
                self.v_C = self.beta * self.v_C + (1 - self.beta) * g_C
                self.cache_C = self.beta * self.cache_C + (1 - self.beta) * (g_C * g_C)
                v_C_hat = self.v_C / (1 - self.beta ** self.t)
                cache_C_hat = self.cache_C / (1 - self.beta ** self.t)
                g_C_hat = v_C_hat / (np.sqrt(cache_C_hat) + epsilon)
            return g_A_hat, g_C_hat
    
        else:
            raise ValueError(f"Unknown optimization method: {self.method}")

def compute_membership_matrix(graph, num_comm):
    membership = np.zeros((len(graph.nodes), num_comm), dtype=float)
    for node_id, node in enumerate(graph.nodes, 0):
        for comm_id in node.communities:
            membership[node_id, comm_id-1] += 1.0/len(node.communities)
    return membership

def compute_loss_matrix(adjacency_matrix, membership_matrix, pairwise_matrix):
    return adjacency_matrix - membership_matrix @ pairwise_matrix @ membership_matrix.T

def compute_frobenius_norm(adjacency_matrix, membership_matrix, pairwise_matrix):
    loss_matrix = compute_loss_matrix(adjacency_matrix, membership_matrix, pairwise_matrix)
    return np.linalg.norm(loss_matrix, "fro")**2

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

def compute_updated_pairwise_matrix(adjacency_matrix, membership_matrix, pairwise_matrix, optimizer):
    raw_gradient_C = 2.0*(membership_matrix.T @ compute_loss_matrix(adjacency_matrix, membership_matrix, pairwise_matrix) @ membership_matrix)
    _, g_C_hat = optimizer.update(None, raw_gradient_C)
    gradient_pairwise_matrix = optimizer.eta * g_C_hat
#    pdb.set_trace()
#    gradient_pairwise_matrix -= 2.0*eta*l2_lambda*pairwise_matrix
#    gradient_pairwise_matrix -= (entropy_lambda*(1+np.log(pairwise_matrix+epsilon)))
#    gradient_pairwise_matrix = gradient_pairwise_matrix/(np.linalg.norm(gradient_pairwise_matrix)+epsilon)
#    change_in_pairwise_matrix = (np.abs(gradient_pairwise_matrix) > delta).any()
    change_in_pairwise_matrix = True
    if change_in_pairwise_matrix:
        pairwise_matrix += gradient_pairwise_matrix
        # Make C symmetric to keep the undirected aspect
        pairwise_matrix = np.maximum(pairwise_matrix, 0)
        pairwise_matrix = (pairwise_matrix+pairwise_matrix.T)/2
    return pairwise_matrix, change_in_pairwise_matrix, np.min(gradient_pairwise_matrix), np.max(gradient_pairwise_matrix)

def compute_updated_membership_matrix(original_membership_matrix, adjacency_matrix, membership_matrix, pairwise_matrix, optimizer):
    raw_gradient_A = 4.0*(compute_loss_matrix(adjacency_matrix, membership_matrix, pairwise_matrix) @ membership_matrix @ pairwise_matrix)
    g_A_hat, _ = optimizer.update(raw_gradient_A, None)
    gradient_membership_matrix = optimizer.eta * g_A_hat
#    pdb.set_trace()
#    gradient_membership_matrix -= 2.0*eta*l2_lambda*membership_matrix
#    gradient_membership_matrix -= (entropy_lambda*(1+np.log(membership_matrix+epsilon)))
#    gradient_membership_matrix = gradient_membership_matrix/(np.linalg.norm(gradient_membership_matrix)+epsilon)
#    change_in_membership_matrix = (np.abs(gradient_membership_matrix) > delta).any()
    change_in_membership_matrix = True
    if change_in_membership_matrix:
        membership_matrix += gradient_membership_matrix
        # normalzie membership_matrix
        membership_matrix = project_membership_matrix(original_membership_matrix, membership_matrix)
    return membership_matrix, change_in_membership_matrix, np.min(gradient_membership_matrix), np.max(gradient_membership_matrix)

def plot_loss_history(loss_history):
    print(f"Loss is a funny thing: {min(loss_history)}, {max(loss_history)}")
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, "b-", linewidth=2)
    plt.ylabel("Loss")
    plt.xlabel("Iteration")
    plt.title("Loss Convergence")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"loss{graph_size}_a.png")
    plt.close()

#l2 for affinity
#use masking to do both l1 and l2 regularization

def iterative_update_membership_pairwise_matrix(adjacency_matrix, membership_matrix, pairwise_matrix):
    optimizer = Optimizer()
    optimizer.initialize(membership_matrix.shape, pairwise_matrix.shape)

    updated_membership_matrix = membership_matrix.copy()
    updated_pairwise_matrix = pairwise_matrix.copy()
    change_in_membership_matrix = change_in_pairwise_matrix = True
    iteration_count = 0
    best_loss = 10000
    cooldown_value = 0
    loss_history = []
    sign_history = []
    while iteration_count < 20000 and (change_in_pairwise_matrix or change_in_membership_matrix):
        updated_membership_matrix, change_in_membership_matrix, min_pair, max_pair = compute_updated_membership_matrix(membership_matrix.copy(), adjacency_matrix.copy(), updated_membership_matrix.copy(), updated_pairwise_matrix.copy(), optimizer)
        updated_pairwise_matrix, change_in_pairwise_matrix, min_mem, max_mem = compute_updated_pairwise_matrix(adjacency_matrix.copy(), updated_membership_matrix.copy(), updated_pairwise_matrix.copy(), optimizer)
        current_loss = compute_frobenius_norm(adjacency_matrix, updated_membership_matrix, updated_pairwise_matrix)
        loss_history.append(current_loss)
        current_sign = compute_loss_matrix(adjacency_matrix, membership_matrix, pairwise_matrix)
        sign_history.append(current_sign)
        if abs(current_loss - best_loss) > 0.05:
            best_loss = current_loss
#            print(f"Iteration {iteration_count}: New best loss: {best_loss:.4f}")
            cooldown_value = 0
        else:
            cooldown_value += 1
        if cooldown_value > 50:
            optimizer.eta *= 0.25
            cooldown_value = 0
            print(f"Iteration {iteration_count}: Loss decreased to {best_loss:.4f}, reducing eta to {optimizer.eta:.6f}")
        if iteration_count % 100 == 0:
            print(f"Iteration {iteration_count}:: Loss: {current_loss:.4f}")
        iteration_count += 1
        if optimizer.eta <= epsilon:
            print(f"Reached eta less than {epsilon}")
            break
    print(f"Final iteration: {iteration_count}")
#    print(f"pairwise_matrix:\n{pairwise_matrix}\nupdated_pairwise_matrix:\n{updated_pairwise_matrix}")
#    print(f"membership_matrix:\n{membership_matrix}\nupdated_membership_matrix:\n{updated_membership_matrix}")
#    plot_loss_history(loss_history)
    return np.round(updated_membership_matrix, 2), np.round(updated_pairwise_matrix, 2)

def prepare_stubs(graph, pairwise_matrix, membership_matrix):
    num_nodes = len(membership_matrix)
    W = membership_matrix @ pairwise_matrix @ membership_matrix.T
    predicted_stubs = {}
    for idx, row in enumerate(W, 1):
        predicted_stubs[idx] = float(sum(row))
    return predicted_stubs

def prepare_default_stubs(graph):
    stubs_per_node = sum(sum(graph.adjacency_matrix))/len(graph.nodes)
    return {key: stubs_per_node for key, _ in enumerate(graph.nodes, 1)}

def check_diff(original_stubs, predicted_stubs):
    incorrect_stubs = sum([abs(round(a-b, 0)) for a, b in zip(original_stubs.values(), predicted_stubs.values())])
    total_stubs = sum([a for a in original_stubs.values()])
    return 50.0*incorrect_stubs/total_edges

def learning_rate_scheduler():
    # CosineAnnealingLR
    # plot LR along with loss
    pass

g = Graph()
num_comm = read_metis_file(g)
pairwise_matrix, total_edges = compute_pairwise_matrix(g, num_comm)
membership_matrix = compute_membership_matrix(g, num_comm)
uniform_predicted_stubs = prepare_default_stubs(g)
updated_membership_matrix, updated_pairwise_matrix = iterative_update_membership_pairwise_matrix(g.adjacency_matrix, membership_matrix, pairwise_matrix)
original_stubs = {}
for i in range(len(g.nodes)):
    original_stubs[i+1] = int(sum(g.adjacency_matrix[:, i]))
original_stubs = dict(sorted(original_stubs.items()))
predicted_stubs = prepare_stubs(g, updated_pairwise_matrix, updated_membership_matrix)
pdb.set_trace()
#print(f"OriginalStubs: {original_stubs}\nPredictedStubs: {predicted_stubs}")
print(f"Percent of incorrect stubs: {check_diff(original_stubs, predicted_stubs)}")
print(f"Percent of incorrect stubs in even division: {check_diff(original_stubs, uniform_predicted_stubs)}")

