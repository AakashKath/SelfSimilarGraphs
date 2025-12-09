import copy
import math
import numpy as np
import random
import time

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

def reduce_unmatched_stubs(unmatched_stubs, open_stubs, src_comm_len):
    for i in range(len(unmatched_stubs)-1, -1, -1):
        v_id, v_size, v_num = unmatched_stubs[i]
        if v_id in open_stubs and v_size == src_comm_len:
            v_num -= 1
            if v_num > 0:
                unmatched_stubs[i] = (v_id, v_size, v_num)
            else:
                unmatched_stubs.pop(i)

def break_prior_edges(graph, unmatched_stubs, possible_nodes, overlap_size, src_comm_len, predicted_edges, predicted_matrix, req_num_dest_nodes):
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

def has_unmatched_entry(unmatched_stubs, u_id, src_comm_len):
    for i, (v_id, v_size, v_num) in enumerate(unmatched_stubs):
        if u_id == v_id and v_size == src_comm_len:
            return True
    return False

def look_for_open_stubs(graph, union_nodes, unmatched_stubs, node, src_comm_len, overlap_size):
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

def find_unmatched_stubs(graph):
    unmatched_stubs = []
    for node_id, node in enumerate(graph.nodes, 1):
        for overlap_size, num_items in node.order.items():
            unmatched_stubs.append((node_id, overlap_size, num_items))
    return unmatched_stubs

def connect_to_chosen_nodes(graph, chosen_nodes, node, node_id, src_comm_len, predicted_edges, predicted_matrix):
    for c_id in chosen_nodes:
        chosen_node = graph.get_node(c_id)
        chosen_node.predicted_neighbors.append(node_id)
        node.predicted_neighbors.append(c_id)
        update_node_order(chosen_node, src_comm_len, 1)
        predicted_edges.append((min(node_id, c_id), max(node_id, c_id)))
        predicted_matrix[node_id-1, c_id-1] = 1
        predicted_matrix[c_id-1, node_id-1] = 1

def find_involved_nodes(graph, node, node_id):
    all_nodes = set.union(*(graph.communities[comm] for comm in node.connectivity)) - {node_id}
    return set([v for v in all_nodes if set(graph.get_node(v).communities).issubset(node.connectivity)])

def update_node_order(node, overlap_size, num_connections):
    node.order[overlap_size] -= num_connections
    if node.order[overlap_size] == 0:
        node.order.pop(overlap_size)

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

def careful_matching(graph, unmatched_stubs, predicted_edges, predicted_matrix, initial_open_stubs):
    while unmatched_stubs:
        if 1.0*sum([v[2] for v in unmatched_stubs])/initial_open_stubs < 0.1:
            return
        node_id, overlap_size, req_num_dest_nodes = unmatched_stubs.pop(0)
        node = graph.get_node(node_id)
        src_comm_len = len(node.communities)
        union_nodes = set.union(*(graph.communities[comm] for comm in node.connectivity)) - set(node.predicted_neighbors) - {node_id}
        open_stubs, possible_nodes = look_for_open_stubs(graph, union_nodes, unmatched_stubs, node, src_comm_len, overlap_size)
        if len(open_stubs) > 0:
            open_stubs = open_stubs[:min(req_num_dest_nodes, len(open_stubs))]
            connect_to_chosen_nodes(graph, open_stubs, node, node_id, src_comm_len, predicted_edges, predicted_matrix)
            reduce_unmatched_stubs(unmatched_stubs, open_stubs, src_comm_len)
            update_node_order(node, overlap_size, len(open_stubs))
            req_num_dest_nodes -= len(open_stubs)
        if req_num_dest_nodes == 0:
            continue
        # If not enough open stubs, break some pre-existing edges
        random.shuffle(possible_nodes)
#        chosen_nodes = list(map(int, np.random.choice(possible_nodes, size=req_num_dest_nodes, replace=False)))
        chosen_nodes = break_prior_edges(graph, unmatched_stubs, possible_nodes, overlap_size, src_comm_len, predicted_edges, predicted_matrix, req_num_dest_nodes)
        connect_to_chosen_nodes(graph, chosen_nodes, node, node_id, src_comm_len, predicted_edges, predicted_matrix)
        update_node_order(node, overlap_size, req_num_dest_nodes)

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
            connect_to_chosen_nodes(graph, chosen_nodes, node, node_id, src_comm_len, predicted_edges, predicted_matrix)
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
        print(f"After Careful Remaining: {sum([v[2] for v in unmatched_stubs])}")
        random_matching(graph, unmatched_stubs, predicted_edges, predicted_matrix)
        time3 = time.time()
        print(f"Time : Random matching is {(time3-time2)/(time2-time1)} times faster than Careful matching.")

    return temp_predicted_edges, temp_predicted_matrix, predicted_edges, predicted_matrix


