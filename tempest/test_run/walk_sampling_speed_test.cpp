#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <iomanip>
#include <fstream>

#include "test_utils.h"
#include "../src/core/helpers.cuh"
#include "../src/proxies/Tempest.cuh"
#include "../test/test_utils.h"

#ifdef HAS_CUDA
constexpr bool DEFAULT_USE_GPU = true;
#else
constexpr bool DEFAULT_USE_GPU = false;
#endif

constexpr bool DEFAULT_IS_DIRECTED = false;
constexpr int DEFAULT_MAX_WALK_LENGTH = 80;
constexpr int DEFAULT_NUM_TOTAL_WALKS = 10'000'000;
constexpr int DEFAULT_NUM_WALKS_PER_NODE = -1;
constexpr auto DEFAULT_EDGE_PICKER = "ExponentialIndex";
constexpr auto DEFAULT_START_PICKER = "Uniform";
constexpr auto DEFAULT_KERNEL_LAUNCH_TYPE_STR = "NODE_GROUPED";
constexpr auto DEFAULT_WALK_DIRECTION_STR = "Forward_In_Time";
// DEFAULT_TIMESCALE_BOUND is defined in common/const.cuh.

void print_walk_performance_stats(
    const size_t num_walks,
    const size_t* walk_lens,
    const double duration_seconds)
{
    if (duration_seconds <= 0.0) {
        std::cout << "Duration too small to compute throughput.\n";
        return;
    }

    size_t total_steps        = 0;
    size_t walks_with_steps   = 0;  // walks with len > 0; the rest died at start.
    for (size_t i = 0; i < num_walks; ++i) {
        total_steps += walk_lens[i];
        if (walk_lens[i] > 0) ++walks_with_steps;
    }
    const size_t dead_at_start = num_walks - walks_with_steps;

    const double walks_per_sec =
        static_cast<double>(num_walks) / duration_seconds;
    const double steps_per_sec =
        static_cast<double>(total_steps) / duration_seconds;
    // Average length over walks that actually advanced.  Walks that died at
    // the start (len == 0) contribute zero steps but would dilute the mean
    // if they stayed in the denominator, hiding the realised walk length of
    // the live cohort and breaking apples-to-apples with engines whose walks
    // always include at least the start vertex (e.g. TEA-reimpl).
    const double avg_walk_length =
        walks_with_steps > 0
            ? static_cast<double>(total_steps) / static_cast<double>(walks_with_steps)
            : 0.0;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Walk generation completed.\n"
              << "Total walks: " << num_walks << "\n"
              << "Total steps: " << total_steps << "\n"
              << "Dead at start: " << dead_at_start << "\n"
              << "Average walk length: " << avg_walk_length
              << "  (over " << walks_with_steps << " walks with len > 0)\n"
              << "Time taken: " << duration_seconds << " seconds\n"
              << "Throughput:\n"
              << "  Walks/sec: " << walks_per_sec << "\n"
              << "  Steps/sec: " << steps_per_sec << "\n";
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage:\n"
                  << argv[0]
                  << " <edge_file_path>"
                  << " [use_gpu (0|1, default=" << (DEFAULT_USE_GPU ? 1 : 0) << ")]"
                  << " [is_directed (0|1, default=0)]"
                  << " [num_total_walks]"
                  << " [num_walks_per_node]"
                  << " [max_walk_length]"
                  << " [edge_picker]"
                  << " [start_picker]"
                  << " [kernel_launch_type=FULL_WALK|NODE_GROUPED|NODE_GROUPED_GLOBAL_ONLY]"
                  << " [walk_dump_file]"
                  << " [timescale_bound (default=" << DEFAULT_TIMESCALE_BOUND << ")]"
                  << " [walk_direction=Forward_In_Time|Backward_In_Time"
                  << " (default=" << DEFAULT_WALK_DIRECTION_STR << ")]"
                  << " [cuda_device_id (default=0; ignored if use_gpu=0)]\n";
        return 1;
    }

    const std::string file_path = argv[1];

    const bool use_gpu =
        (argc > 2)
            ? (std::stoi(argv[2]) != 0)
            : DEFAULT_USE_GPU;

    const bool is_directed =
        (argc > 3)
            ? (std::stoi(argv[3]) != 0)
            : DEFAULT_IS_DIRECTED;

    const int num_total_walks =
        (argc > 4) ? std::stoi(argv[4]) : DEFAULT_NUM_TOTAL_WALKS;

    const int num_walks_per_node =
        (argc > 5) ? std::stoi(argv[5]) : DEFAULT_NUM_WALKS_PER_NODE;

    const int max_walk_length =
        (argc > 6) ? std::stoi(argv[6]) : DEFAULT_MAX_WALK_LENGTH;

    const std::string edge_picker =
        (argc > 7) ? argv[7] : DEFAULT_EDGE_PICKER;

    const std::string start_picker =
        (argc > 8) ? argv[8] : DEFAULT_START_PICKER;

    const std::string kernel_launch_type_str =
        (argc > 9) ? argv[9] : DEFAULT_KERNEL_LAUNCH_TYPE_STR;

    const std::string walk_dump_file =
        (argc > 10) ? argv[10] : "";

    const double timescale_bound =
        (argc > 11) ? std::stod(argv[11]) : DEFAULT_TIMESCALE_BOUND;

    const std::string walk_direction_str =
        (argc > 12) ? argv[12] : DEFAULT_WALK_DIRECTION_STR;

    const WalkDirection walk_direction =
        walk_direction_from_string(walk_direction_str);

    const int cuda_device_id =
        (argc > 13) ? std::stoi(argv[13]) : 0;

    std::cout << "Running on: " << (use_gpu ? "GPU" : "CPU") << "\n";
    if (use_gpu) {
        std::cout << "CUDA device: " << cuda_device_id << "\n";
    }
    std::cout << "Graph type: "
              << (is_directed ? "Directed" : "Undirected") << "\n";
    std::cout << "Kernel launch type: " << kernel_launch_type_str << "\n";
    std::cout << "Edge picker: " << edge_picker
              << "  Start picker: " << start_picker << "\n";
    std::cout << "Timescale bound: " << timescale_bound << "\n";
    std::cout << "Walk direction: " << walk_direction_str << "\n";

    const auto edge_infos = read_edges_from_csv(file_path);

    auto [sources, targets, timestamps] =
        convert_edge_tuples_to_components(edge_infos);

    std::cout << "Edges loaded: " << edge_infos.size() << "\n";

    const RandomPickerType edge_picker_enum =
        picker_type_from_string(edge_picker);

    const RandomPickerType start_picker_enum =
        picker_type_from_string(start_picker);

    const KernelLaunchType kernel_launch_type =
        kernel_launch_type_from_string(kernel_launch_type_str);

    const bool enable_temporal_node2vec =
        edge_picker_enum == RandomPickerType::TemporalNode2Vec;
    const bool enable_weight_computation =
        edge_picker_enum  == RandomPickerType::ExponentialWeight
        || start_picker_enum == RandomPickerType::ExponentialWeight;

    Tempest walker(
        is_directed,
        use_gpu,
        /*max_time_capacity=*/-1,
        enable_weight_computation,
        enable_temporal_node2vec,
        timescale_bound,
        /*node2vec_p=*/DEFAULT_NODE2VEC_P,
        /*node2vec_q=*/DEFAULT_NODE2VEC_Q,
        /*walk_padding_value=*/EMPTY_NODE_VALUE,
        /*global_seed=*/EMPTY_GLOBAL_SEED,
        /*shuffle_walk_order=*/DEFAULT_SHUFFLE_WALK_ORDER,
        cuda_device_id
    );

    std::cout << "\nIngesting edges in bulk...\n";

    const auto ingest_start =
        std::chrono::high_resolution_clock::now();

    walker.add_multiple_edges(
        sources.data(),
        targets.data(),
        timestamps.data(),
        timestamps.size()
    );

    const auto ingest_end =
        std::chrono::high_resolution_clock::now();

    const double ingest_seconds =
        std::chrono::duration<double>(ingest_end - ingest_start).count();

    std::cout << std::fixed << std::setprecision(6)
              << "Ingest time: " << ingest_seconds << " seconds\n";
    std::cout << "Graph constructed. Nodes: "
              << walker.get_node_count()
              << " Edges: "
              << walker.get_edge_count() << "\n";

    std::cout << "\nGenerating walks...\n";

    const auto walk_start =
        std::chrono::high_resolution_clock::now();

    auto walks_with_edge_features =
        (num_walks_per_node == -1)
            ? walker.get_random_walks_and_times(
                max_walk_length,
                &edge_picker_enum,
                num_total_walks,
                &start_picker_enum,
                walk_direction,
                kernel_launch_type)
            : walker.get_random_walks_and_times_for_all_nodes(
                max_walk_length,
                &edge_picker_enum,
                num_walks_per_node,
                &start_picker_enum,
                walk_direction,
                kernel_launch_type);

    const auto walk_end =
        std::chrono::high_resolution_clock::now();

    const double walk_seconds =
        std::chrono::duration<double>(walk_end - walk_start).count();

    std::cout << std::fixed << std::setprecision(6)
              << "Walk time: " << walk_seconds << " seconds\n";

    const auto& walk_set = walks_with_edge_features.walk_set;
    print_walk_performance_stats(
        walk_set.num_walks(),
        walk_set.walk_lens_ptr(),
        walk_seconds);

    if (!walk_dump_file.empty()) {
        dump_walks_to_file(
            walk_set,
            max_walk_length,
            walk_dump_file);
    }

    const size_t memory_footprint_bytes =
        walker.get_memory_used() + walk_set.get_memory_used();
    const double memory_gb =
        static_cast<double>(memory_footprint_bytes) / (1024.0 * 1024.0 * 1024.0);
    std::cout << "Memory used (GB): " << memory_gb << std::endl;

    return 0;
}
