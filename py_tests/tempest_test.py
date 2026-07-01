import csv
import time

from tempest import Tempest


GPU_USAGE_MODE = False

def print_walks_for_nodes(nodes, timestamps, lens, limit=100):
    for idx in range(min(len(lens), limit)):
        walk_nodes = nodes[idx, :lens[idx]]
        walk_timestamps = timestamps[idx, :lens[idx]]
        print(f"Walk {idx}: {','.join(map(str, walk_nodes))}, Timestamps: {','.join(map(str, walk_timestamps))}")

if __name__ == '__main__':
    tempest_obj = Tempest(
        is_directed=False,
        use_gpu=GPU_USAGE_MODE,
        max_time_capacity=-1,
        enable_weight_computation=False
    )

    data_file_path = '../data/sample_data.csv'

    sources = []
    targets = []
    timestamps = []

    with open(data_file_path, mode='r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=",")
        next(csv_reader)
        for row in csv_reader:
            u = int(row[0])
            i = int(row[1])
            ts = int(row[2])
            sources.append(u)
            targets.append(i)
            timestamps.append(ts)

    print(f"--- Total edges: {len(timestamps)} ---")

    start_time = time.time()
    tempest_obj.add_multiple_edges(sources, targets, timestamps)
    print(f"--- Edge addition time : {time.time() - start_time}---")

    print(f'Total edges: {tempest_obj.get_edge_count()}')
    print(f'Total nodes: {tempest_obj.get_node_count()}')

    start_time = time.time()
    walks_1 = tempest_obj.get_random_walks_and_times(
        max_walk_len=20,
        walk_bias="ExponentialIndex",
        num_walks_total=100_000,
        initial_edge_bias="Uniform",
        walk_direction="Forward_In_Time"
    )

    walks_2 = tempest_obj.get_random_walks_and_times(
        max_walk_len=20,
        walk_bias="ExponentialIndex",
        num_walks_total=100_000,
        initial_edge_bias="Uniform",
        walk_direction="Forward_In_Time"
    )
    print(f"--- Walk generation time : {time.time() - start_time}---")

    print(f"Number of walks {len(walks_2[0])}")
    print_walks_for_nodes(walks_2[0][:100], walks_2[1][:100], walks_2[2][:100])
