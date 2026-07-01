#ifndef DATA_TEMPORAL_GRAPH_VIEW_CUH
#define DATA_TEMPORAL_GRAPH_VIEW_CUH

#include <cstddef>
#include <cstdint>

#include "../common/macros.cuh"
#include "temporal_graph_data.cuh"

// non-owning POD view of TemporalGraphData for pass-by-value to kernels.
struct TemporalGraphView {
    bool    is_directed;
    bool    enable_temporal_node2vec;
    int64_t latest_timestamp;
    double  inv_p;
    double  inv_q;

    const int*     sources;
    const int*     targets;
    const int64_t* timestamps;
    size_t         num_edges;

    const size_t*  timestamp_group_offsets;
    const int64_t* unique_timestamps;
    size_t         num_groups;

    const int*     active_node_ids;
    size_t         active_node_ids_size;
    int            max_node_id;

    const double*  forward_cumulative_weights_exponential;
    size_t         forward_cumulative_weights_exponential_size;
    const double*  backward_cumulative_weights_exponential;
    size_t         backward_cumulative_weights_exponential_size;

    const size_t*  node_adj_offsets;
    size_t         node_adj_offsets_size;
    const int*     node_adj_neighbors;
    size_t         node_adj_neighbors_size;

    const size_t*  node_group_outbound_offsets;
    const size_t*  node_group_inbound_offsets;

    const size_t*  node_ts_sorted_outbound_indices;
    size_t         node_ts_sorted_outbound_indices_size;
    const size_t*  node_ts_sorted_inbound_indices;
    size_t         node_ts_sorted_inbound_indices_size;

    const size_t*  count_ts_group_per_node_outbound;
    const size_t*  count_ts_group_per_node_inbound;

    const size_t*  node_ts_group_outbound_offsets;
    const size_t*  node_ts_group_inbound_offsets;

    const double*  outbound_forward_cumulative_weights_exponential;
    size_t         outbound_forward_cumulative_weights_exponential_size;
    const double*  outbound_backward_cumulative_weights_exponential;
    size_t         outbound_backward_cumulative_weights_exponential_size;
    const double*  inbound_backward_cumulative_weights_exponential;
    size_t         inbound_backward_cumulative_weights_exponential_size;
};

// view aliases into data; invalid if data is mutated or destroyed.
HOST inline TemporalGraphView make_temporal_graph_view(
    const TemporalGraphData& data) {
    TemporalGraphView v{};
    v.is_directed              = data.is_directed;
    v.enable_temporal_node2vec = data.enable_temporal_node2vec;
    v.latest_timestamp         = data.latest_timestamp;
    v.inv_p                    = data.inv_p;
    v.inv_q                    = data.inv_q;

    v.sources           = data.sources.data();
    v.targets           = data.targets.data();
    v.timestamps        = data.timestamps.data();
    v.num_edges         = data.timestamps.size();

    v.timestamp_group_offsets = data.timestamp_group_offsets.data();
    v.unique_timestamps       = data.unique_timestamps.data();
    v.num_groups              = data.unique_timestamps.size();

    v.active_node_ids      = data.active_node_ids.data();
    v.active_node_ids_size = data.active_node_ids.size();
    v.max_node_id          = data.max_node_id;

    v.forward_cumulative_weights_exponential =
        data.forward_cumulative_weights_exponential.data();
    v.forward_cumulative_weights_exponential_size =
        data.forward_cumulative_weights_exponential.size();
    v.backward_cumulative_weights_exponential =
        data.backward_cumulative_weights_exponential.data();
    v.backward_cumulative_weights_exponential_size =
        data.backward_cumulative_weights_exponential.size();

    v.node_adj_offsets        = data.node_adj_offsets.data();
    v.node_adj_offsets_size   = data.node_adj_offsets.size();
    v.node_adj_neighbors      = data.node_adj_neighbors.data();
    v.node_adj_neighbors_size = data.node_adj_neighbors.size();

    v.node_group_outbound_offsets = data.node_group_outbound_offsets.data();
    v.node_group_inbound_offsets  = data.node_group_inbound_offsets.data();

    v.node_ts_sorted_outbound_indices = data.node_ts_sorted_outbound_indices.data();
    v.node_ts_sorted_outbound_indices_size =
        data.node_ts_sorted_outbound_indices.size();
    v.node_ts_sorted_inbound_indices  = data.node_ts_sorted_inbound_indices.data();
    v.node_ts_sorted_inbound_indices_size =
        data.node_ts_sorted_inbound_indices.size();

    v.count_ts_group_per_node_outbound = data.count_ts_group_per_node_outbound.data();
    v.count_ts_group_per_node_inbound  = data.count_ts_group_per_node_inbound.data();

    v.node_ts_group_outbound_offsets = data.node_ts_group_outbound_offsets.data();
    v.node_ts_group_inbound_offsets  = data.node_ts_group_inbound_offsets.data();

    v.outbound_forward_cumulative_weights_exponential =
        data.outbound_forward_cumulative_weights_exponential.data();
    v.outbound_forward_cumulative_weights_exponential_size =
        data.outbound_forward_cumulative_weights_exponential.size();
    v.outbound_backward_cumulative_weights_exponential =
        data.outbound_backward_cumulative_weights_exponential.data();
    v.outbound_backward_cumulative_weights_exponential_size =
        data.outbound_backward_cumulative_weights_exponential.size();
    v.inbound_backward_cumulative_weights_exponential =
        data.inbound_backward_cumulative_weights_exponential.data();
    v.inbound_backward_cumulative_weights_exponential_size =
        data.inbound_backward_cumulative_weights_exponential.size();

    return v;
}

#endif // DATA_TEMPORAL_GRAPH_VIEW_CUH
