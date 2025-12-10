import argparse
import matplotlib.pyplot as plt
import numpy as np
import pdb
import random
from collections import defaultdict, deque, Counter
from graph import GraphReader
from itertools import repeat
from math import ceil
from mpl_toolkits.mplot3d import Axes3D

class CommunityObject:
    comm_id: int
    stubs: list[int]
    community_size: int
    overlaps: list[list[int]]

    def __init__(self, comm_id, stubs, community_size):
        self.comm_id = comm_id
        self.stubs = stubs
        self.community_size = community_size
        self.overlaps = []

class CommunityList:
    idx: int
    communities: list[CommunityObject]

    def __init__(self):
        self.idx = 1
        self.communities = []

    def get_community(self, comm_id):
        return self.communities[comm_id-1]

    def add_community(self, stubs, community_size):
        comm = CommunityObject(self.idx, stubs, community_size)
        self.communities.append(comm)
        self.idx += 1
        return comm

    def break_some_bonds(self, stub_dict, by_weight, weight):
        counter = 0
        while stub_dict:
            stub_list = by_weight[weight]
            b_id = random.choice(list(set(stub_list)-set(stub_dict.keys())))
            broken_comm = self.get_community(b_id)
            b_ids = random.choice([v for v in broken_comm.overlaps if len(v) == weight])
            for b_id in b_ids:
                broken_comm = self.get_community(b_id)
                broken_comm.overlaps.remove(b_ids)
                if b_id not in stub_dict:
                    stub_dict[b_id] = 0
                stub_dict[b_id] += 1
            stub_dict = self.match_stub_dict(stub_dict, weight)

    def match_stub_dict(self, stub_dict, weight):
        while True:
            keys = np.array(list(stub_dict.keys()))
            if len(keys) < weight:
                if len(keys) != 0:
                    return {k: v for k, v in stub_dict.items() if v != 0}
                return {}
            values = np.array(list(stub_dict.values()), dtype=float)
            prob = values/values.sum()
            chosen_comms = list(map(int, np.random.choice(keys, size=weight, replace=False, p=prob)))
            for comm_id in chosen_comms:
                stub_dict[comm_id] -= 1
                if stub_dict[comm_id] == 0:
                    del stub_dict[comm_id]
                comm = self.get_community(comm_id)
                comm.overlaps.append(chosen_comms)

    def stub_matching(self):
        by_weight = defaultdict(list)
        for comm in self.communities:
            for w in comm.stubs:
                by_weight[w].append(comm.comm_id)
        overlaps = []
        unmatched_stubs = []
        for weight, stub_list in by_weight.items():
            stub_dict = dict(Counter(stub_list))
            result = self.match_stub_dict(stub_dict, weight)
            if result:
                unmatched_stubs.append((weight, result))
        print(f"Percent of unmatched stubs: {100.0*sum([len(c) for _, c in unmatched_stubs])/sum([len(v) for v in by_weight.values()])}")
        # Phase 2: Randomly connect rest of the stubs
        while unmatched_stubs:
            weight, stub_dict = unmatched_stubs.pop()
            self.break_some_bonds(stub_dict, by_weight, weight)
        print(f"Percent of unmatched stubs: {100.0*sum([len(c) for _, c in unmatched_stubs])/sum([len(v) for v in by_weight.values()])}")
        return overlaps

def plot_3d(details, ax, title):
    xs, ys, zs = [], [], []
    for key, y in details.items():
        x, z = map(int, key.split("_"))
        xs.append(x)
        ys.append(y)
        zs.append(z)

    ax.scatter(xs, ys, zs, c="red", s=60)

    ax.set_xlabel("Size of Community")
    ax.set_ylabel("Number of Communities")
    if title == "details_with_size":
        ax.set_zlabel("size of overlaps")
    else:
        ax.set_zlabel("number of overlaps")
    ax.set_title(title)

def update_plot_dict(dictionary, key):
    if key not in dictionary:
        dictionary[key] = 0
    dictionary[key] += 1

def draw_community_graph(graph):
    details_with_size = {} # Will be used for node distribution
    details_with_number = {} # Will be used for stub matching
    max_overlap_size = 0
    for comm_id, node_list in graph.communities.items():
        number_of_overlap = set()
        size_of_overlap = 0
        for node_id in node_list:
            node = graph.get_node(node_id)
            number_of_overlap.update(node.communities)
            max_overlap_size = max(max_overlap_size, len(node.communities))
            if len(node.communities) > 1:
                size_of_overlap += 1
        update_plot_dict(details_with_size, f"{len(node_list)}_{size_of_overlap}")
        update_plot_dict(details_with_number, f"{len(node_list)}_{len(number_of_overlap)-1}")

    pdb.set_trace()
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    plot_3d(details_with_size, ax1, "details_with_size")
    plot_3d(details_with_number, ax2, "details_with_number")
    plt.tight_layout()
    plt.show()

def collect_graph_info(graph):
    collapsed_info = {}
    for comm_id, node_list in graph.communities.items():
        empty_dict = {}
        empty_set = set()
        number_of_overlap = 0
        for node_id in node_list:
            node = graph.get_node(node_id)
            comm_combination = "_".join(map(str, sorted(node.communities)))
            if comm_combination not in empty_set:
                empty_set.add(comm_combination)
                combination_count = len(node.communities)
                empty_dict[combination_count] = empty_dict.get(combination_count, 0) + 1
                number_of_overlap += 1
        # TODO: _{comm_id} is appended to keep communities different, should be removed once we implement scale
        key = f"{len(node_list)}_{number_of_overlap}_{comm_id}"
        if key not in collapsed_info:
            collapsed_info[key] = {}
        for k, v in empty_dict.items():
            collapsed_info[key][k] = collapsed_info[key].get(k, 0) + v
    return collapsed_info

def draw_graph_with_collapsed_info(collapsed_info, scale):
    original_communities = sum([sum(list(v.values()))//int(k.split('_')[1]) for k, v in collapsed_info.items()])
    stub_distribution = []
    weights = []
    for ck, cv in collapsed_info.items():
        number_of_unique_overlap = int(ck.split("_")[1])
        number_of_communities = sum(cv.values())//number_of_unique_overlap
        stub_distribution.append((list(cv.keys()), list(cv.values()), number_of_unique_overlap, int(ck.split("_")[0])))
        weights.append(number_of_communities)
    # TODO: Remove the debug part to include scale, but before that please include scale logic at stub level
    comm_list = CommunityList()
    debug = True
    if debug:
        sampled_comm = [x for v, r in zip(stub_distribution, weights) for x in repeat(v, r)]
        for stubs, weight, required_stubs, community_size in sampled_comm:
            comm_list.add_community([x for v, r in zip(stubs, weight) for x in repeat(v, r)], community_size)
    else:
        required_comm = ceil(original_communities*scale/100)
        sampled_comm = random.choices(stub_distribution, weights, k=required_comm)
        for stubs, weights, required_stubs, community_size in sampled_comm:
            sampled_stubs = random.choices(stubs, weights, k=required_stubs)
            comm_list.add_community(sampled_stubs, community_size)
    overlaps = comm_list.stub_matching()

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
    args = parser.parse_args()
    reader = GraphReader(args.community_size, args.child_directory)
    graph = reader.read_metis_file()
#    draw_community_graph(graph)
    collapsed_info = collect_graph_info(graph)
    draw_graph_with_collapsed_info(collapsed_info, 100)

