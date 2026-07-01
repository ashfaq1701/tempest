#ifndef NODE_GROUPED_KERNELS_COMMON_CUH
#define NODE_GROUPED_KERNELS_COMMON_CUH

#include "../../../common/macros.cuh"
#include "../../../common/cuda_config.cuh"
#include "../../../data/temporal_graph_view.cuh"
#include "../../../data/walk_set/walk_set_view.cuh"
#include "../../../random/pickers.cuh"
#include "../../../utils/random.cuh"
#include "../../../utils/utils.cuh"

namespace tempest {

#ifdef HAS_CUDA

// matches get_node_edge_at_device direction resolution
struct NodeDirPtrs {
    const size_t* count_ts_group_per_node;
    const size_t* node_ts_groups_offsets;
    const size_t* node_ts_sorted_indices;
    const size_t* node_edge_offsets;
    const double* weights;
    size_t        weights_size;
};

template <bool IsDirected, bool Forward>
HOST DEVICE __forceinline__ const size_t*
count_ts_group_per_node_for_dir(const TemporalGraphView& view) {
    return Forward ? view.count_ts_group_per_node_outbound
                   : (IsDirected ? view.count_ts_group_per_node_inbound
                                 : view.count_ts_group_per_node_outbound);
}

template <bool IsDirected, bool Forward>
DEVICE __forceinline__ NodeDirPtrs
resolve_node_dir_ptrs(const TemporalGraphView& view) {
    NodeDirPtrs p;

    p.count_ts_group_per_node =
        Forward ? view.count_ts_group_per_node_outbound
                : (IsDirected ? view.count_ts_group_per_node_inbound
                              : view.count_ts_group_per_node_outbound);

    p.node_ts_groups_offsets =
        Forward ? view.node_ts_group_outbound_offsets
                : (IsDirected ? view.node_ts_group_inbound_offsets
                              : view.node_ts_group_outbound_offsets);

    p.node_ts_sorted_indices =
        Forward ? view.node_ts_sorted_outbound_indices
                : (IsDirected ? view.node_ts_sorted_inbound_indices
                              : view.node_ts_sorted_outbound_indices);

    p.node_edge_offsets =
        Forward ? view.node_group_outbound_offsets
                : (IsDirected ? view.node_group_inbound_offsets
                              : view.node_group_outbound_offsets);

    p.weights =
        Forward ? view.outbound_forward_cumulative_weights_exponential
                : (IsDirected ? view.inbound_backward_cumulative_weights_exponential
                              : view.outbound_backward_cumulative_weights_exponential);

    p.weights_size =
        Forward ? view.outbound_forward_cumulative_weights_exponential_size
                : (IsDirected ? view.inbound_backward_cumulative_weights_exponential_size
                              : view.outbound_backward_cumulative_weights_exponential_size);

    return p;
}

template <RandomPickerType PickerType>
HOST DEVICE constexpr inline int coop_block_smem_g_cap() {
    return random_pickers::is_index_based_picker(PickerType)
               ? G_THRESHOLD_BLOCK_INDEX
               : G_THRESHOLD_BLOCK_WEIGHT;
}

template <RandomPickerType PickerType>
HOST DEVICE constexpr inline int coop_warp_smem_g_cap() {
    return random_pickers::is_index_based_picker(PickerType)
               ? G_THRESHOLD_WARP_INDEX
               : G_THRESHOLD_WARP_WEIGHT;
}

// shared per-walk tail for the four coop kernels
template <bool IsDirected, bool Forward>
DEVICE __forceinline__ void sample_edge_and_add_hop(
    const TemporalGraphView& view,
    WalkSetView              walk_set,
    const NodeDirPtrs&       ptrs,
    const size_t*            offsets_slice,
    const long               local_pos,
    const int                G,
    const size_t             node_edge_end,
    const int                node_id,
    const size_t             walk_idx,
    const double             r_edge) {

    const size_t valid_edge_start = offsets_slice[local_pos];
    const size_t valid_edge_end =
        (local_pos + 1 < G)
            ? offsets_slice[local_pos + 1]
            : node_edge_end;
    if (valid_edge_start >= valid_edge_end) return;

    const long edge_idx = static_cast<long>(ptrs.node_ts_sorted_indices[
        valid_edge_start +
        generate_random_number_bounded_by(
            static_cast<int>(valid_edge_end - valid_edge_start),
            r_edge)]);

    if constexpr (IsDirected) {
        walk_set.add_hop(walk_idx,
                         Forward ? view.targets[edge_idx]
                                 : view.sources[edge_idx],
                         view.timestamps[edge_idx],
                         edge_idx);
    } else {
        const int next_node = pick_other_number(
            view.sources[edge_idx], view.targets[edge_idx], node_id);
        walk_set.add_hop(walk_idx, next_node,
                         view.timestamps[edge_idx], edge_idx);
    }
}

#endif // HAS_CUDA

} // namespace tempest

#endif // NODE_GROUPED_KERNELS_COMMON_CUH
