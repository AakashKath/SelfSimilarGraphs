from collections import defaultdict

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


