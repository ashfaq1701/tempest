#ifndef WALK_STEP_HELPERS_CUH
#define WALK_STEP_HELPERS_CUH

#include "../common/macros.cuh"
#include "../data/temporal_graph_view.cuh"

namespace temporal_graph {

    HOST DEVICE inline double get_group_exponential_weight_from_cumulative(
        const double* weights,
        const size_t current_group_pos,
        const size_t node_group_begin) {
        if (current_group_pos == node_group_begin) {
            return weights[current_group_pos];
        }
        return weights[current_group_pos] - weights[current_group_pos - 1];
    }

    template<bool Forward, bool IsDirected>
    HOST DEVICE size_t get_node_group_edge_end(
        const TemporalGraphView& view,
        const int node_id,
        const size_t* node_ts_groups_offsets,
        const size_t group_pos,
        const size_t node_group_end) {

        if (group_pos + 1 < node_group_end) {
            return node_ts_groups_offsets[group_pos + 1];
        }

        if constexpr (Forward) {
            return view.node_group_outbound_offsets[node_id + 1];
        }

        return IsDirected
            ? view.node_group_inbound_offsets[node_id + 1]
            : view.node_group_outbound_offsets[node_id + 1];
    }

    template<bool Forward, bool IsDirected>
    HOST DEVICE int get_candidate_node(
        const TemporalGraphView& view,
        const int node_id,
        const size_t edge_idx) {

        const int src = view.sources[edge_idx];
        const int dst = view.targets[edge_idx];

        if constexpr (IsDirected) {
            return Forward ? dst : src;
        }

        return (src == node_id) ? dst : src;
    }

}

#endif // WALK_STEP_HELPERS_CUH
