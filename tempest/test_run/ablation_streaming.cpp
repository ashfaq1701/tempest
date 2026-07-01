#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif

#include "../src/common/nvtx.cuh"
#include "../src/proxies/Tempest.cuh"
#include "../test/test_utils.h"
#include "test_utils.h"

#ifdef HAS_CUDA
constexpr bool DEFAULT_USE_GPU = true;
#else
constexpr bool DEFAULT_USE_GPU = false;
#endif


RandomPickerType parse_picker(const std::string &s) {
    if (s == "uniform") return RandomPickerType::Uniform;
    if (s == "linear") return RandomPickerType::Linear;
    if (s == "exponential_index") return RandomPickerType::ExponentialIndex;
    if (s == "exponential_weight") return RandomPickerType::ExponentialWeight;
    throw std::runtime_error("Invalid picker type");
}

KernelLaunchType parse_kernel_launch_type(const std::string &s) {
    if (s == "full_walk") return KernelLaunchType::FULL_WALK;
    if (s == "node_grouped") return KernelLaunchType::NODE_GROUPED;
    if (s == "node_grouped_global_only") return KernelLaunchType::NODE_GROUPED_GLOBAL_ONLY;
    throw std::runtime_error(
        "Invalid kernel launch type — expected one of: "
        "full_walk, node_grouped, node_grouped_global_only");
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <file_path>"
                  << " [use_gpu=1]"
                  << " [picker=exponential_index]"
                  << " [kernel_launch_type=full_walk|node_grouped|node_grouped_global_only]"
                  << " [is_directed=0]"
                  << " [num_walks_per_node=1]"
                  << " [num_batches=5]"
                  << " [num_windows=3]"
                  << " [max_walk_len=80]"
                  << " [timescale_bound=-1]"
                  << " [block_dim=256]"
                  << " [w_threshold_warp=" << W_THRESHOLD_WARP << "]\n";
        return 1;
    }

    const std::string file_path = argv[1];
    const bool use_gpu = (argc > 2) ? std::stoi(argv[2]) : DEFAULT_USE_GPU;
    const std::string picker_str = (argc > 3) ? argv[3] : "exponential_index";
    const std::string kernel_launch_type_str = (argc > 4) ? argv[4] : "full_walk";
    const bool is_directed = (argc > 5) ? std::stoi(argv[5]) != 0 : false;
    const int num_walks_per_node = (argc > 6) ? std::stoi(argv[6]) : 1;
    const int num_batches = (argc > 7) ? std::stoi(argv[7]) : 5;
    const int num_windows = (argc > 8) ? std::stoi(argv[8]) : 3;
    const int max_walk_len = (argc > 9) ? std::stoi(argv[9]) : 80;
    const double timescale_bound = (argc > 10) ? std::stod(argv[10]) : -1;
    const size_t block_dim = (argc > 11) ? static_cast<size_t>(std::stoi(argv[11])) : 256;
    const int w_threshold_warp = (argc > 12) ? std::stoi(argv[12])
                                              : static_cast<int>(W_THRESHOLD_WARP);

    const RandomPickerType hop_picker = parse_picker(picker_str);
    const KernelLaunchType kernel_launch_type =
        parse_kernel_launch_type(kernel_launch_type_str);
    constexpr RandomPickerType start_picker = RandomPickerType::Uniform;

    std::cout << "=== Streaming Benchmark (decoupled batch & window) ===\n"
              << "File: " << file_path << "\n"
              << "Device: " << (use_gpu ? "GPU" : "CPU") << "\n"
              << "Hop picker: " << picker_str << "\n"
              << "Kernel launch type: " << kernel_launch_type_str << "\n"
              << "Directed graph: " << (is_directed ? "yes" : "no") << "\n"
              << "Walks per node: " << num_walks_per_node << "\n"
              << "Num batches: " << num_batches << "\n"
              << "Num windows: " << num_windows << "\n"
              << "Max walk length: " << max_walk_len << "\n"
              << "Timescale bound: " << timescale_bound << "\n"
              << "Block dim: " << block_dim << "\n"
              << "W threshold (solo): " << w_threshold_warp << "\n";

    std::vector<int> sources, targets;
    std::vector<int64_t> timestamps;
    {
        NvtxRange r("edge_load");
        const auto edge_infos = read_edges_from_csv(file_path, -1);
        std::tie(sources, targets, timestamps) =
            convert_edge_tuples_to_components(edge_infos);
        std::cout << "Edges loaded: " << edge_infos.size() << "\n";
    }

    auto [min_it, max_it] =
        std::minmax_element(timestamps.begin(), timestamps.end());
    const int64_t min_ts = *min_it;
    const int64_t max_ts = *max_it;

    const int64_t batch_duration =
        (max_ts - min_ts) / std::max(1, num_batches);
    const int64_t window_duration =
        (max_ts - min_ts) / std::max(1, num_windows);

    std::cout << "Batch duration Δ_batch = " << batch_duration << "\n"
              << "Window duration Δ_window = " << window_duration << "\n";

    const bool use_weight = hop_picker == RandomPickerType::ExponentialWeight;

    Tempest trw(
        is_directed,
        use_gpu,
        window_duration,
        use_weight,
        false,
        timescale_bound
    );

    std::vector<size_t> order(timestamps.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b) {
                  return timestamps[a] < timestamps[b];
              });

    std::vector<double> ingestion_times;
    std::vector<double> walk_times;
    size_t total_walks = 0;
    double total_walk_len_sum = 0.0;

    size_t cursor = 0;
    const size_t N = timestamps.size();

    if (num_batches > 0) {
        size_t warm_cursor = 0;
        const int64_t warm_end_ts =
            (num_batches == 1) ? max_ts + 1 : min_ts + batch_duration;

        std::vector<int> warm_src, warm_dst;
        std::vector<int64_t> warm_ts;
        while (warm_cursor < N && timestamps[order[warm_cursor]] < warm_end_ts) {
            const size_t idx = order[warm_cursor++];
            warm_src.push_back(sources[idx]);
            warm_dst.push_back(targets[idx]);
            warm_ts.push_back(timestamps[idx]);
        }

        NvtxRange r("warmup");
        trw.add_multiple_edges(
            warm_src.data(), warm_dst.data(), warm_ts.data(), warm_ts.size());
        auto warm_walks = trw.get_random_walks_and_times_for_all_nodes(
            max_walk_len, &hop_picker, num_walks_per_node, &start_picker,
            WalkDirection::Forward_In_Time, kernel_launch_type, block_dim,
            w_threshold_warp);
#ifdef HAS_CUDA
        if (use_gpu) cudaDeviceSynchronize();
#endif
        std::cout << "\n[Warmup] Ingested " << warm_src.size()
                  << " edges, sampled " << warm_walks.walk_set.size()
                  << " walks (discarded).\n";
    }

    for (int b = 0; b < num_batches; ++b) {
        const int64_t batch_end_ts =
            (b == num_batches - 1)
                ? max_ts + 1
                : min_ts + (b + 1) * batch_duration;

        std::vector<int> batch_src, batch_dst;
        std::vector<int64_t> batch_ts;

        while (cursor < N && timestamps[order[cursor]] < batch_end_ts) {
            const size_t idx = order[cursor++];
            batch_src.push_back(sources[idx]);
            batch_dst.push_back(targets[idx]);
            batch_ts.push_back(timestamps[idx]);
        }

        std::cout << "\n[Batch " << b + 1 << "] Edges: "
                  << batch_src.size() << "\n";

        double ingest_time = 0.0;
        {
            NvtxRange r("ingestion_batch");
            const auto t0 = std::chrono::high_resolution_clock::now();
            trw.add_multiple_edges(
                batch_src.data(),
                batch_dst.data(),
                batch_ts.data(),
                batch_ts.size());

#ifdef HAS_CUDA
            if (use_gpu) cudaDeviceSynchronize();
#endif

            const auto t1 = std::chrono::high_resolution_clock::now();
            ingest_time =
                std::chrono::duration<double>(t1 - t0).count();
        }
        ingestion_times.push_back(ingest_time);

        double walk_time = 0.0;
        size_t walks_this_batch = 0;
        double avg_len_batch = 0.0;

        {
            NvtxRange r("walk_sampling_batch");
            const auto t0 = std::chrono::high_resolution_clock::now();

            const auto walks_with_edge_feats = trw.get_random_walks_and_times_for_all_nodes(
                max_walk_len,
                &hop_picker,
                num_walks_per_node,
                &start_picker,
                WalkDirection::Forward_In_Time,
                kernel_launch_type,
                block_dim,
                w_threshold_warp);

            #ifdef HAS_CUDA
            if (use_gpu) cudaDeviceSynchronize();
            #endif

            const auto t1 = std::chrono::high_resolution_clock::now();
            walk_time =
                std::chrono::duration<double>(t1 - t0).count();

            walks_this_batch = walks_with_edge_feats.walk_set.size();
            avg_len_batch = get_average_walk_length(walks_with_edge_feats.walk_set);

            total_walks += walks_this_batch;
            total_walk_len_sum += avg_len_batch * walks_this_batch;
        }
        walk_times.push_back(walk_time);

        std::cout << "  Ingest time: " << ingest_time << " sec\n"
                  << "  Walk time:   " << walk_time << " sec\n"
                  << "  Walks:       " << walks_this_batch << "\n"
                  << "  Avg length:  " << avg_len_batch << "\n";
    }

    const double total_ingestion =
        std::accumulate(ingestion_times.begin(), ingestion_times.end(), 0.0);
    const double total_walk =
        std::accumulate(walk_times.begin(), walk_times.end(), 0.0);
    const double final_avg_len =
        (total_walks > 0) ? (total_walk_len_sum / total_walks) : 0.0;

    std::cout << "\n=== Summary ===\n"
              << "Block dim:            " << block_dim << "\n"
              << "W threshold (solo):   " << w_threshold_warp << "\n"
              << "Total ingestion time: " << total_ingestion << " sec\n"
              << "Mean ingestion time:  "
              << (total_ingestion / num_batches) << " sec\n"
              << "Total walk time:      " << total_walk << " sec\n"
              << "Mean walk time/batch: "
              << (total_walk / num_batches) << " sec\n"
              << "Total walks:          " << total_walks << "\n"
              << "Final avg walk length:" << final_avg_len << "\n"
              << "Throughput:           "
              << (total_walks / total_walk) << " walks/sec\n"
              << "Steps/sec:            "
              << (total_walk_len_sum / total_walk) << " steps/sec\n";

    return 0;
}
