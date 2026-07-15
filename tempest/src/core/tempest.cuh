#ifndef TEMPEST_STORE_H
#define TEMPEST_STORE_H

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <vector>

#include "../common/macros.cuh"
#include "../common/const.cuh"
#include "../common/cuda_config.cuh"
#include "../data/structs.cuh"
#include "../data/enums.cuh"
#include "../data/buffer.cuh"
#include "../data/temporal_graph_data.cuh"
#include "../data/walk_set/walks_with_edge_features_host.cuh"
#include "../graph/temporal_graph.cuh"
#include "../graph/node_features.cuh"

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace core {

class Tempest {
public:
    Tempest(
        bool is_directed,
        bool use_gpu,
        int64_t max_time_capacity = -1,
        bool enable_weight_computation = false,
        bool enable_temporal_node2vec = false,
        double timescale_bound = DEFAULT_TIMESCALE_BOUND,
        double node2vec_p = DEFAULT_NODE2VEC_P,
        double node2vec_q = DEFAULT_NODE2VEC_Q,
        int walk_padding_value = EMPTY_NODE_VALUE,
        uint64_t global_seed = EMPTY_GLOBAL_SEED,
        bool shuffle_walk_order = DEFAULT_SHUFFLE_WALK_ORDER,
        // CUDA device index for the instance when use_gpu=true. Ignored when
        // use_gpu=false. Default 0 preserves single-GPU behavior; pass a
        // non-zero value to pin this instance to a specific GPU while
        // another library (e.g. PyTorch) holds a different one.
        int cuda_device_id = 0);

    ~Tempest();

    Tempest(const Tempest&) = delete;
    Tempest& operator=(const Tempest&) = delete;
    // moves manage stream ownership manually
    Tempest(Tempest&&) noexcept;
    Tempest& operator=(Tempest&&) noexcept;

    TemporalGraphData&       data()       { return data_; }
    const TemporalGraphData& data() const { return data_; }

    int      walk_padding_value() const { return walk_padding_value_; }
    uint64_t global_seed()        const { return global_seed_; }
    bool     shuffle_walk_order() const { return shuffle_walk_order_; }
    bool     is_directed()        const { return data_.is_directed; }

    Buffer<int>&       last_batch_unique_sources()       { return last_batch_unique_sources_; }
    const Buffer<int>& last_batch_unique_sources() const { return last_batch_unique_sources_; }
    Buffer<int>&       last_batch_unique_targets()       { return last_batch_unique_targets_; }
    const Buffer<int>& last_batch_unique_targets() const { return last_batch_unique_targets_; }

#ifdef HAS_CUDA
    const cudaDeviceProp& cuda_device_prop() const { return cuda_device_prop_; }

    // CUDA device this instance is pinned to (use_gpu=true). Undefined
    // when use_gpu=false.
    int cuda_device_id() const { return cuda_device_id_; }

    // per-instance non-blocking stream; avoids serializing on legacy stream 0
    cudaStream_t stream() const { return stream_; }

    void sync_stream() const {
        if (data_.use_gpu && stream_ != nullptr) {
            cudaStreamSynchronize(stream_);
        }
    }
#endif

    void add_multiple_edges(
        const int* sources, const int* targets,
        const int64_t* timestamps, size_t num_edges,
        const float* edge_features = nullptr, size_t feature_dim = 0,
        size_t block_dim = BLOCK_DIM);

    void add_multiple_edges(
        const std::vector<std::tuple<int, int, int64_t>>& edges,
        const float* edge_features = nullptr, size_t feature_dim = 0,
        size_t block_dim = BLOCK_DIM);

    WalksWithEdgeFeaturesHost get_random_walks_and_times_for_all_nodes(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias = nullptr,
        WalkDirection walk_direction = WalkDirection::Forward_In_Time,
        KernelLaunchType kernel_launch_type = DEFAULT_KERNEL_LAUNCH_TYPE,
        size_t block_dim = BLOCK_DIM,
        int w_threshold_warp = W_THRESHOLD_WARP);

    WalksWithEdgeFeaturesHost get_random_walks_and_times_for_last_batch(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias = nullptr,
        WalkDirection walk_direction = WalkDirection::Forward_In_Time,
        KernelLaunchType kernel_launch_type = DEFAULT_KERNEL_LAUNCH_TYPE,
        size_t block_dim = BLOCK_DIM,
        int w_threshold_warp = W_THRESHOLD_WARP);

    WalksWithEdgeFeaturesHost get_random_walks_and_times_for_nodes(
        const int* seed_nodes,
        size_t num_seed_nodes,
        const int64_t* cutoff_times,
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias = nullptr,
        WalkDirection walk_direction = WalkDirection::Forward_In_Time,
        KernelLaunchType kernel_launch_type = DEFAULT_KERNEL_LAUNCH_TYPE,
        size_t block_dim = BLOCK_DIM,
        int w_threshold_warp = W_THRESHOLD_WARP);

    WalksWithEdgeFeaturesHost get_random_walks_and_times(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_total,
        const RandomPickerType* initial_edge_bias = nullptr,
        WalkDirection walk_direction = WalkDirection::Forward_In_Time,
        KernelLaunchType kernel_launch_type = DEFAULT_KERNEL_LAUNCH_TYPE,
        size_t block_dim = BLOCK_DIM,
        int w_threshold_warp = W_THRESHOLD_WARP);

    void set_node_features(
        const int* node_ids, size_t num_nodes,
        const float* node_features_src, size_t feature_dim);

    size_t get_node_count() const;
    size_t get_edge_count() const;
    std::vector<int> get_node_ids() const;
    std::vector<int64_t> get_node_degrees(
        const int* nodes, size_t n,
        WalkDirection direction = WalkDirection::Forward_In_Time) const;

    // Per-node latest edge timestamp strictly before each node's cutoff (-1 if none),
    // and per-node #edges strictly before the cutoff. cutoff_times = nullptr => whole
    // history for every node; else one int64 cutoff per input node.
    std::vector<int64_t> get_latest_timestamps_for_nodes(
        const int* nodes, size_t n, const int64_t* cutoff_times,
        WalkDirection direction = WalkDirection::Forward_In_Time) const;
    std::vector<int64_t> get_node_participation_counts(
        const int* nodes, size_t n, const int64_t* cutoff_times,
        WalkDirection direction = WalkDirection::Forward_In_Time) const;

    std::vector<Edge> get_edges() const;
    bool get_is_directed() const { return data_.is_directed; }
    void clear();
    size_t get_memory_used() const;

private:
    TemporalGraphData data_;
    int      walk_padding_value_;
    uint64_t global_seed_;
    bool     shuffle_walk_order_;
    Buffer<int> last_batch_unique_sources_{/*use_gpu=*/false};
    Buffer<int> last_batch_unique_targets_{/*use_gpu=*/false};
#ifdef HAS_CUDA
    cudaDeviceProp cuda_device_prop_{};
    cudaStream_t   stream_{nullptr};
    int            cuda_device_id_{0};
#endif
};

} // namespace core

namespace tempest {

    HOST void add_multiple_edges(
        core::Tempest* trw,
        const int* sources, const int* targets,
        const int64_t* timestamps, size_t num_edges,
        const float* edge_features = nullptr, size_t feature_dim = 0,
        size_t block_dim = BLOCK_DIM);

    HOST size_t get_node_count(const core::Tempest* trw);
    HOST size_t get_edge_count(const core::Tempest* trw);
    HOST std::vector<int>  get_node_ids(const core::Tempest* trw);
    HOST std::vector<int64_t> get_node_degrees(
        const core::Tempest* trw,
        const int* nodes, size_t n, WalkDirection direction);

    // Per-node cutoff-bounded queries — separate CPU (_std) / GPU (_cuda) impls.
    HOST std::vector<int64_t> get_latest_timestamps_for_nodes_std(
        const core::Tempest* trw, const int* nodes, size_t n,
        const int64_t* cutoff_times, WalkDirection direction);
    HOST std::vector<int64_t> get_node_participation_counts_std(
        const core::Tempest* trw, const int* nodes, size_t n,
        const int64_t* cutoff_times, WalkDirection direction);
    #ifdef HAS_CUDA
    HOST std::vector<int64_t> get_latest_timestamps_for_nodes_cuda(
        const core::Tempest* trw, const int* nodes, size_t n,
        const int64_t* cutoff_times, WalkDirection direction);
    HOST std::vector<int64_t> get_node_participation_counts_cuda(
        const core::Tempest* trw, const int* nodes, size_t n,
        const int64_t* cutoff_times, WalkDirection direction);
    #endif

    HOST std::vector<Edge> get_edges(const core::Tempest* trw);
    HOST bool              get_is_directed(const core::Tempest* trw);
    HOST void              clear(core::Tempest* trw);

    HOST WalksWithEdgeFeaturesHost get_random_walks_and_times_for_all_nodes_std(
        core::Tempest* trw,
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias = nullptr,
        WalkDirection walk_direction = WalkDirection::Forward_In_Time);

    HOST WalksWithEdgeFeaturesHost get_random_walks_and_times_for_last_batch_std(
        core::Tempest* trw,
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias = nullptr,
        WalkDirection walk_direction = WalkDirection::Forward_In_Time);

    HOST WalksWithEdgeFeaturesHost get_random_walks_and_times_for_nodes_std(
        core::Tempest* trw,
        const int* seed_nodes,
        size_t num_seed_nodes,
        const int64_t* cutoff_times,
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias = nullptr,
        WalkDirection walk_direction = WalkDirection::Forward_In_Time);

    HOST WalksWithEdgeFeaturesHost get_random_walks_and_times_std(
        core::Tempest* trw,
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_total,
        const RandomPickerType* initial_edge_bias = nullptr,
        WalkDirection walk_direction = WalkDirection::Forward_In_Time);

#ifdef HAS_CUDA
    HOST WalksWithEdgeFeaturesHost get_random_walks_and_times_for_all_nodes_cuda(
        core::Tempest* trw,
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias = nullptr,
        WalkDirection walk_direction = WalkDirection::Forward_In_Time,
        KernelLaunchType kernel_launch_type = DEFAULT_KERNEL_LAUNCH_TYPE,
        size_t block_dim = BLOCK_DIM,
        int w_threshold_warp = W_THRESHOLD_WARP);

    HOST WalksWithEdgeFeaturesHost get_random_walks_and_times_for_last_batch_cuda(
        core::Tempest* trw,
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias = nullptr,
        WalkDirection walk_direction = WalkDirection::Forward_In_Time,
        KernelLaunchType kernel_launch_type = DEFAULT_KERNEL_LAUNCH_TYPE,
        size_t block_dim = BLOCK_DIM,
        int w_threshold_warp = W_THRESHOLD_WARP);

    HOST WalksWithEdgeFeaturesHost get_random_walks_and_times_for_nodes_cuda(
        core::Tempest* trw,
        const int* seed_nodes,
        size_t num_seed_nodes,
        const int64_t* cutoff_times,
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias = nullptr,
        WalkDirection walk_direction = WalkDirection::Forward_In_Time,
        KernelLaunchType kernel_launch_type = DEFAULT_KERNEL_LAUNCH_TYPE,
        size_t block_dim = BLOCK_DIM,
        int w_threshold_warp = W_THRESHOLD_WARP);

    HOST WalksWithEdgeFeaturesHost get_random_walks_and_times_cuda(
        core::Tempest* trw,
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_total,
        const RandomPickerType* initial_edge_bias = nullptr,
        WalkDirection walk_direction = WalkDirection::Forward_In_Time,
        KernelLaunchType kernel_launch_type = DEFAULT_KERNEL_LAUNCH_TYPE,
        size_t block_dim = BLOCK_DIM,
        int w_threshold_warp = W_THRESHOLD_WARP);
#endif

    HOST size_t get_memory_used(const core::Tempest* trw);
}

#endif // TEMPEST_STORE_H
