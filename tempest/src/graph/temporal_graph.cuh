#ifndef TEMPORAL_GRAPH_CUH
#define TEMPORAL_GRAPH_CUH

#include <cstddef>
#include <vector>

#include "../common/macros.cuh"
#include "../common/const.cuh"
#include "../common/error_handlers.cuh"
#include "../data/structs.cuh"
#include "../data/temporal_graph_data.cuh"
#include "../data/temporal_graph_view.cuh"
#include "../data/buffer.cuh"

#include "edge_data.cuh"
#include "node_edge_index.cuh"
#include "node_features.cuh"

namespace temporal_graph {

    HOST void update_temporal_weights(TemporalGraphData& data);

    HOST DEVICE size_t get_total_edges(const TemporalGraphData& data);

    HOST size_t get_node_count(const TemporalGraphData& data);

    HOST int64_t get_latest_timestamp(const TemporalGraphData& data);

    HOST std::vector<int> get_node_ids(const TemporalGraphData& data);

    HOST std::vector<int64_t> get_node_degrees(
        const TemporalGraphData& data,
        const int* nodes,
        size_t n,
        bool forward);

    HOST std::vector<Edge> get_edges(const TemporalGraphData& data);

    HOST void sort_and_merge_edges_std(
        TemporalGraphData& data,
        size_t start_idx);

    HOST void delete_old_edges_std(TemporalGraphData& data);

    HOST void add_multiple_edges_std(
        TemporalGraphData& data,
        const int* sources,
        const int* targets,
        const int64_t* timestamps,
        size_t num_new_edges,
        const float* edge_features = nullptr,
        size_t feature_dim = 0);

    HOST size_t count_timestamps_less_than_std(
        const TemporalGraphData& data, int64_t timestamp);

    HOST size_t count_timestamps_greater_than_std(
        const TemporalGraphData& data, int64_t timestamp);

    HOST size_t count_node_timestamps_less_than_std(
        const TemporalGraphData& data, int node_id, int64_t timestamp);

    HOST size_t count_node_timestamps_greater_than_std(
        const TemporalGraphData& data, int node_id, int64_t timestamp);

    #ifdef HAS_CUDA

    HOST void sort_and_merge_edges_cuda(
        TemporalGraphData& data,
        size_t start_idx);

    HOST void delete_old_edges_cuda(TemporalGraphData& data);

    HOST void add_multiple_edges_cuda(
        TemporalGraphData& data,
        const int* sources,
        const int* targets,
        const int64_t* timestamps,
        size_t num_new_edges,
        const float* edge_features = nullptr,
        size_t feature_dim = 0);

    HOST size_t count_timestamps_less_than_cuda(
        const TemporalGraphData& data, int64_t timestamp);

    HOST size_t count_timestamps_greater_than_cuda(
        const TemporalGraphData& data, int64_t timestamp);

    HOST size_t count_node_timestamps_less_than_cuda(
        const TemporalGraphData& data, int node_id, int64_t timestamp);

    HOST size_t count_node_timestamps_greater_than_cuda(
        const TemporalGraphData& data, int node_id, int64_t timestamp);

    #endif

    HOST size_t get_memory_used(const TemporalGraphData& data);

}

#endif // TEMPORAL_GRAPH_CUH
