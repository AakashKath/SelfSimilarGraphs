import argparse
import matplotlib.pyplot as plt
import pdb
import random
from collections import defaultdict, deque
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

    def stub_matching(self):
        by_weight = defaultdict(list)
        for comm in self.communities:
            for w in comm.stubs:
                by_weight[w].append(comm.comm_id)
        overlaps = []
        unmatched_stubs = []
        for weight, stub_list in by_weight.items():
            stub_list = deque(stub_list)
            random.shuffle(stub_list)
            counter = 0
            while counter < len(stub_list):
                combination = set()
                while len(combination) != weight and counter < len(stub_list):
                    comm_id = stub_list.popleft()
                    if comm_id in combination:
                        stub_list.append(comm_id)
                        counter += 1
                        continue
                    combination.add(comm_id)
                    counter = 0
                if len(combination) == weight:
                    overlaps.append(list(combination))
                    for comm_id in combination:
                        comm = self.get_community(comm_id)
                        comm.overlaps.append(combination)
                else:
                    unmatched_stubs.append((weight, combination))
        print(f"Percent of unmatched stubs: {100.0*sum([len(c) for _, c in unmatched_stubs])/sum([len(v) for v in by_weight.values()])}")
        # TODO: Maybe we need- Phase 2: Randomly connect rest of the stubs
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
            comm_combination = "_".join(map(str, node.communities))
            if comm_combination not in empty_set:
                empty_set.add(comm_combination)
                combination_count = len(node.communities)
                empty_dict[combination_count] = empty_dict.get(combination_count, 0) + 1
                number_of_overlap += 1
        key = f"{len(node_list)}_{number_of_overlap}"
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
    debug = True
    if debug:
        sampled_comm = [x for v, r in zip(stub_distribution, weights) for x in repeat(v, r)]
    else:
        required_comm = ceil(original_communities*scale/100)
        sampled_comm = random.choices(stub_distribution, weights, k=required_comm)
    comm_list = CommunityList()
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

