#ifndef DATA_TEMPORAL_GRAPH_DATA_CUH
#define DATA_TEMPORAL_GRAPH_DATA_CUH

#include <cstddef>
#include <cstdint>

#include "../common/const.cuh"
#include "../common/macros.cuh"
#include "buffer.cuh"

struct TemporalGraphData {
    bool use_gpu = false;

    bool    is_directed               = false;
    int64_t max_time_capacity         = 0;
    int64_t latest_timestamp          = 0;
    double  timescale_bound           = DEFAULT_TIMESCALE_BOUND;
    double  node2vec_p                = DEFAULT_NODE2VEC_P;
    double  node2vec_q                = DEFAULT_NODE2VEC_Q;
    double  inv_p                     = 1.0;
    double  inv_q                     = 1.0;
    bool    enable_weight_computation = false;
    bool    enable_temporal_node2vec  = false;

    Buffer<int>     sources;
    Buffer<int>     targets;
    Buffer<int64_t> timestamps;

    Buffer<float>   edge_features;
    size_t          feature_dim = 0;

    Buffer<size_t>  timestamp_group_offsets;
    Buffer<int64_t> unique_timestamps;

    Buffer<int>     active_node_ids;
    int             max_node_id = -1;

    Buffer<double>  forward_cumulative_weights_exponential;
    Buffer<double>  backward_cumulative_weights_exponential;

    Buffer<size_t>  node_adj_offsets;
    Buffer<int>     node_adj_neighbors;

    Buffer<size_t>  node_group_outbound_offsets;
    Buffer<size_t>  node_group_inbound_offsets;

    Buffer<size_t>  node_ts_sorted_outbound_indices;
    Buffer<size_t>  node_ts_sorted_inbound_indices;

    Buffer<size_t>  count_ts_group_per_node_outbound;
    Buffer<size_t>  count_ts_group_per_node_inbound;

    Buffer<size_t>  node_ts_group_outbound_offsets;
    Buffer<size_t>  node_ts_group_inbound_offsets;

    Buffer<double>  outbound_forward_cumulative_weights_exponential;
    Buffer<double>  outbound_backward_cumulative_weights_exponential;
    Buffer<double>  inbound_backward_cumulative_weights_exponential;

    Buffer<float>   node_features;
    size_t          node_feature_dim = 0;

    TemporalGraphData() = default;

    explicit TemporalGraphData(const bool use_gpu_arg)
        : use_gpu(use_gpu_arg),
          sources(use_gpu_arg),
          targets(use_gpu_arg),
          timestamps(use_gpu_arg),
          edge_features(false),
          timestamp_group_offsets(use_gpu_arg),
          unique_timestamps(use_gpu_arg),
          active_node_ids(use_gpu_arg),
          forward_cumulative_weights_exponential(use_gpu_arg),
          backward_cumulative_weights_exponential(use_gpu_arg),
          node_adj_offsets(use_gpu_arg),
          node_adj_neighbors(use_gpu_arg),
          node_group_outbound_offsets(use_gpu_arg),
          node_group_inbound_offsets(use_gpu_arg),
          node_ts_sorted_outbound_indices(use_gpu_arg),
          node_ts_sorted_inbound_indices(use_gpu_arg),
          count_ts_group_per_node_outbound(use_gpu_arg),
          count_ts_group_per_node_inbound(use_gpu_arg),
          node_ts_group_outbound_offsets(use_gpu_arg),
          node_ts_group_inbound_offsets(use_gpu_arg),
          outbound_forward_cumulative_weights_exponential(use_gpu_arg),
          outbound_backward_cumulative_weights_exponential(use_gpu_arg),
          inbound_backward_cumulative_weights_exponential(use_gpu_arg),
          node_features(false)
    {}

    TemporalGraphData(const TemporalGraphData&) = delete;
    TemporalGraphData& operator=(const TemporalGraphData&) = delete;
    TemporalGraphData(TemporalGraphData&&) noexcept = default;
    TemporalGraphData& operator=(TemporalGraphData&&) noexcept = default;

    ~TemporalGraphData() = default;
};

#endif // DATA_TEMPORAL_GRAPH_DATA_CUH
