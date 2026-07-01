#include <filesystem>
#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>

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

constexpr int     DEFAULT_TOTAL_MINUTES      = 60;
constexpr int     DEFAULT_MINUTES_PER_STEP   = 3;
constexpr int64_t DEFAULT_WINDOW_MS          = 1'800'000;
constexpr int     DEFAULT_NUM_WALKS_PER_NODE = 20;
constexpr int     DEFAULT_MAX_WALK_LEN       = 100;

RandomPickerType parse_picker(const std::string& s) {
    if (s == "uniform")            return RandomPickerType::Uniform;
    if (s == "linear")             return RandomPickerType::Linear;
    if (s == "exponential_index")  return RandomPickerType::ExponentialIndex;
    if (s == "exponential_weight") return RandomPickerType::ExponentialWeight;
    throw std::runtime_error("Invalid picker type: " + s);
}

KernelLaunchType parse_kernel_launch_type(const std::string& s) {
    if (s == "full_walk")                return KernelLaunchType::FULL_WALK;
    if (s == "node_grouped")             return KernelLaunchType::NODE_GROUPED;
    if (s == "node_grouped_global_only") return KernelLaunchType::NODE_GROUPED_GLOBAL_ONLY;
    throw std::runtime_error(
        "Invalid kernel launch type — expected one of: "
        "full_walk, node_grouped, node_grouped_global_only");
}

struct EdgeBatch {
    std::vector<int>     src;
    std::vector<int>     dst;
    std::vector<int64_t> ts;
};

// parquet wins if both exist
static std::pair<std::string, bool>
resolve_shard_path(const std::string& dataset_dir, int minute) {
    const std::string stem = dataset_dir + "/data_" + std::to_string(minute);
    const std::string pq = stem + ".parquet";
    const std::string csv = stem + ".csv";
    if (std::filesystem::exists(pq)) return {pq, true};
    if (std::filesystem::exists(csv)) return {csv, false};
    throw std::runtime_error(
        "no data_" + std::to_string(minute) + ".{parquet,csv} under " + dataset_dir);
}

EdgeBatch load_step(const std::string& dataset_dir,
                    int minute_begin,
                    int minutes_per_step,
                    int total_minutes) {
    EdgeBatch batch;
    for (int j = 0; j < minutes_per_step && (minute_begin + j) < total_minutes; ++j) {
        auto [path, is_parquet] = resolve_shard_path(dataset_dir, minute_begin + j);
        const auto edges = is_parquet
            ? load_edges_from_parquet(path.c_str())
            : read_edges_from_csv(path);
        batch.src.reserve(batch.src.size() + edges.size());
        batch.dst.reserve(batch.dst.size() + edges.size());
        batch.ts.reserve(batch.ts.size() + edges.size());
        for (const auto& e : edges) {
            batch.src.push_back(std::get<0>(e));
            batch.dst.push_back(std::get<1>(e));
            batch.ts.push_back(std::get<2>(e));
        }
    }
    return batch;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr
            << "Usage: " << argv[0] << " <dataset_dir>"
            << " [use_gpu=1]"
            << " [picker=exponential_index]"
            << " [kernel_launch_type=full_walk|node_grouped|node_grouped_global_only]"
            << " [num_walks_per_node=" << DEFAULT_NUM_WALKS_PER_NODE << "]"
            << " [minutes_per_step=" << DEFAULT_MINUTES_PER_STEP << "]"
            << " [window_ms=" << DEFAULT_WINDOW_MS << "]"
            << " [max_walk_len=" << DEFAULT_MAX_WALK_LEN << "]"
            << " [total_minutes=" << DEFAULT_TOTAL_MINUTES << "]"
            << " [timescale_bound=-1]"
            << " [w_threshold_warp=" << W_THRESHOLD_WARP << "]"
            << " [per_batch_csv=]"
            << " [walk_direction=backward|forward]\n";
        return 1;
    }

    const std::string dataset_dir         = argv[1];
    const bool        use_gpu             = (argc > 2) ? std::stoi(argv[2]) != 0 : DEFAULT_USE_GPU;
    const std::string picker_str          = (argc > 3) ? argv[3] : "exponential_index";
    const std::string klt_str             = (argc > 4) ? argv[4] : "node_grouped";
    const int         num_walks_per_node  = (argc > 5) ? std::stoi(argv[5]) : DEFAULT_NUM_WALKS_PER_NODE;
    const int         minutes_per_step    = (argc > 6) ? std::stoi(argv[6]) : DEFAULT_MINUTES_PER_STEP;
    const int64_t     window_ms           = (argc > 7) ? std::stoll(argv[7]) : DEFAULT_WINDOW_MS;
    const int         max_walk_len        = (argc > 8) ? std::stoi(argv[8]) : DEFAULT_MAX_WALK_LEN;
    const int         total_minutes       = (argc > 9) ? std::stoi(argv[9]) : DEFAULT_TOTAL_MINUTES;
    const double      timescale_bound     = (argc > 10) ? std::stod(argv[10]) : -1.0;
    const int         w_threshold_warp    = (argc > 11) ? std::stoi(argv[11])
                                                        : static_cast<int>(W_THRESHOLD_WARP);
    const std::string per_batch_csv       = (argc > 12) ? argv[12] : std::string();
    const std::string walk_dir_str        = (argc > 13) ? argv[13] : "backward";

    const RandomPickerType hop_picker = parse_picker(picker_str);
    const KernelLaunchType kernel_launch_type = parse_kernel_launch_type(klt_str);
    constexpr RandomPickerType start_picker = RandomPickerType::Uniform;
    WalkDirection walk_direction;
    if (walk_dir_str == "backward" || walk_dir_str == "Backward_In_Time") {
        walk_direction = WalkDirection::Backward_In_Time;
    } else if (walk_dir_str == "forward" || walk_dir_str == "Forward_In_Time") {
        walk_direction = WalkDirection::Forward_In_Time;
    } else {
        throw std::runtime_error("Invalid walk_direction — expected 'forward' or 'backward'");
    }

    std::cout << "=== Alibaba streaming benchmark (C++) ===\n"
              << "Dataset dir:       " << dataset_dir << "\n"
              << "Device:            " << (use_gpu ? "GPU" : "CPU") << "\n"
              << "Hop picker:        " << picker_str << "\n"
              << "Kernel type:       " << klt_str << "\n"
              << "Walks per node:    " << num_walks_per_node << "\n"
              << "Minutes per step:  " << minutes_per_step << "\n"
              << "Window (ms):       " << window_ms << "\n"
              << "Max walk len:      " << max_walk_len << "\n"
              << "Total minutes:     " << total_minutes << "\n"
              << "Timescale bound:   " << timescale_bound << "\n"
              << "W threshold (solo):" << w_threshold_warp << "\n"
              << "Per-batch CSV:     " << (per_batch_csv.empty() ? "(none)" : per_batch_csv) << "\n"
              << "Walk direction:    " << walk_dir_str << "\n";

    const bool use_weight = (hop_picker == RandomPickerType::ExponentialWeight);

    Tempest trw(
        /*is_directed=*/true,
        /*use_gpu=*/use_gpu,
        /*max_time_capacity=*/window_ms,
        /*enable_weight_computation=*/use_weight,
        /*enable_temporal_node2vec=*/false,
        /*timescale_bound=*/timescale_bound);

    {
        NvtxRange r("warmup");
        EdgeBatch warm = load_step(dataset_dir, 0, minutes_per_step, total_minutes);
        trw.add_multiple_edges(
            warm.src.data(), warm.dst.data(), warm.ts.data(), warm.ts.size());
        auto warm_walks = trw.get_random_walks_and_times_for_last_batch(
            max_walk_len, &hop_picker, num_walks_per_node, &start_picker,
            walk_direction, kernel_launch_type,
            BLOCK_DIM, w_threshold_warp);
#ifdef HAS_CUDA
        if (use_gpu) cudaDeviceSynchronize();
#endif
        std::cout << "\n[Warmup] Ingested " << warm.src.size()
                  << " edges, sampled " << warm_walks.walk_set.size()
                  << " walks (discarded).\n";
    }

    std::vector<double> ingestion_times;
    std::vector<double> walk_times;
    std::vector<size_t> walks_per_step;
    size_t total_walks = 0;
    double total_walk_len_sum = 0.0;
    size_t total_edges_added  = 0;

    std::vector<std::vector<std::string>> per_batch_rows;
    const std::vector<std::string> per_batch_header = {
        "step_idx", "step_min_begin", "minutes_per_step",
        "edges_in_batch", "ingest_time_sec", "walk_time_sec",
        "walks", "avg_walk_length", "active_edges", "total_edges_added",
        "walks_per_sec", "steps_per_sec",
    };
    auto fmt_d = [](double v) {
        std::ostringstream os; os.precision(9); os << v; return os.str();
    };

    for (int i = 0; i < total_minutes; i += minutes_per_step) {
        EdgeBatch batch = load_step(dataset_dir, i, minutes_per_step, total_minutes);
        total_edges_added += batch.src.size();

        std::cout << "\n[Step t=" << i << "..+" << minutes_per_step
                  << "min] Edges: " << batch.src.size() << "\n";

        double ingest_time = 0.0;
        {
            NvtxRange r("ingestion_batch");
            const auto t0 = std::chrono::high_resolution_clock::now();
            trw.add_multiple_edges(
                batch.src.data(), batch.dst.data(),
                batch.ts.data(), batch.ts.size());
#ifdef HAS_CUDA
            if (use_gpu) cudaDeviceSynchronize();
#endif
            const auto t1 = std::chrono::high_resolution_clock::now();
            ingest_time = std::chrono::duration<double>(t1 - t0).count();
        }
        ingestion_times.push_back(ingest_time);

        double walk_time = 0.0;
        size_t walks_this_step = 0;
        double avg_len_this_step = 0.0;
        {
            NvtxRange r("walk_sampling_batch");
            const auto t0 = std::chrono::high_resolution_clock::now();
            const auto walks = trw.get_random_walks_and_times_for_last_batch(
                max_walk_len, &hop_picker, num_walks_per_node, &start_picker,
                walk_direction, kernel_launch_type,
                BLOCK_DIM, w_threshold_warp);
#ifdef HAS_CUDA
            if (use_gpu) cudaDeviceSynchronize();
#endif
            const auto t1 = std::chrono::high_resolution_clock::now();
            walk_time = std::chrono::duration<double>(t1 - t0).count();

            walks_this_step    = walks.walk_set.size();
            avg_len_this_step  = get_average_walk_length(walks.walk_set);
            total_walks        += walks_this_step;
            total_walk_len_sum += avg_len_this_step * walks_this_step;
        }
        walk_times.push_back(walk_time);
        walks_per_step.push_back(walks_this_step);

        const size_t active_edges = trw.get_edge_count();

        std::cout << "  Ingest time: " << ingest_time << " sec\n"
                  << "  Walk time:   " << walk_time << " sec\n"
                  << "  Walks:       " << walks_this_step << "\n"
                  << "  Avg length:  " << avg_len_this_step << "\n"
                  << "  Active edges in TRW: " << active_edges << "\n"
                  << "  Total edges added:   " << total_edges_added << "\n";

        const double walks_ps_batch = (walk_time > 0.0)
            ? (static_cast<double>(walks_this_step) / walk_time) : 0.0;
        const double steps_ps_batch = (walk_time > 0.0)
            ? (avg_len_this_step * static_cast<double>(walks_this_step) / walk_time)
            : 0.0;
        per_batch_rows.push_back({
            std::to_string(static_cast<int>(walk_times.size())),
            std::to_string(i),
            std::to_string(minutes_per_step),
            std::to_string(batch.src.size()),
            fmt_d(ingest_time),
            fmt_d(walk_time),
            std::to_string(walks_this_step),
            fmt_d(avg_len_this_step),
            std::to_string(active_edges),
            std::to_string(total_edges_added),
            fmt_d(walks_ps_batch),
            fmt_d(steps_ps_batch),
        });
    }

    write_strings_to_csv(per_batch_csv, per_batch_header, per_batch_rows);

    const size_t num_steps = walk_times.size();
    const double total_ingestion =
        std::accumulate(ingestion_times.begin(), ingestion_times.end(), 0.0);
    const double total_walk =
        std::accumulate(walk_times.begin(), walk_times.end(), 0.0);
    const double final_avg_len =
        (total_walks > 0) ? (total_walk_len_sum / static_cast<double>(total_walks)) : 0.0;
    const double inv_steps = num_steps > 0 ? 1.0 / static_cast<double>(num_steps) : 0.0;
    const double walks_per_sec =
        (total_walk > 0.0) ? (static_cast<double>(total_walks) / total_walk) : 0.0;
    const double steps_per_sec =
        (total_walk > 0.0) ? (total_walk_len_sum / total_walk) : 0.0;

    std::cout << "\n=== Summary ===\n"
              << "Steps measured:       " << num_steps << "\n"
              << "Total ingestion time: " << total_ingestion << " sec\n"
              << "Mean ingestion time:  " << (total_ingestion * inv_steps) << " sec\n"
              << "Total walk time:      " << total_walk << " sec\n"
              << "Mean walk time/step:  " << (total_walk * inv_steps) << " sec\n"
              << "Total walks:          " << total_walks << "\n"
              << "Final avg walk length:" << final_avg_len << "\n"
              << "Throughput:           " << walks_per_sec << " walks/sec\n"
              << "Steps/sec:            " << steps_per_sec << " steps/sec\n";

    return 0;
}
