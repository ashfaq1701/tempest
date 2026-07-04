#ifndef NODE_GROUPED_KERNELS_COMMON_CUH
#define NODE_GROUPED_KERNELS_COMMON_CUH

#include "../../../common/macros.cuh"
#include "../../../common/cuda_config.cuh"
#include "../../../data/temporal_graph_view.cuh"
#include "../../../data/walk_set/walk_set_view.cuh"
#include "../../../random/pickers.cuh"
#include "../../../utils/random.cuh"
#include "../../../utils/utils.cuh"
// find_group_pos_slice / filter_valid_groups_by_timestamp_slice and the
// node2vec rejection sampler (pulls in temporal_node2vec_helpers.cuh).
#include "../../../graph/edge_selectors.cuh"

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

// Append the hop for a chosen global edge_idx, resolving the landing node by
// direction. One home for the add_hop so the plain proposal and the node2vec
// rejection path stay identical.
template <bool IsDirected, bool Forward>
DEVICE __forceinline__ void add_edge_hop(
    const TemporalGraphView& view,
    WalkSetView              walk_set,
    const int                node_id,
    const long               edge_idx,
    const size_t             walk_idx) {

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

    add_edge_hop<IsDirected, Forward>(view, walk_set, node_id, edge_idx, walk_idx);
}

// Per-walk group-pick + edge-sample, shared by all four coop kernels. Callers
// pass either the smem-preloaded structures (first_ts / s_cum_weights set,
// sorted_indices / view_timestamps null) or the global ones (the mirror). The
// non-node2vec path is byte-identical to the previous inline
// find_group_pos_slice + sample_edge_and_add_hop.
//
// node2vec cannot ride the cooperative group-CDF: beta depends on the walk's
// own prev_node, so each walk must sample independently. For prev_node != -1 we
// restrict to the timestamp/cutoff-valid slice and delegate to the SAME
// rejection sampler the solo / full-walk paths use (reads group/weight tables
// from global via ptrs; the smem preload is simply bypassed here — this is the
// paper's per-node-dispatch fallback for node2vec, made correct). prev_node is
// read from walk history, so the first hop out of a walk's second node is
// biased by its first node, matching every other path. prev_node == -1 (only
// the very first hop) falls through to the plain proposal, which for
// TemporalNode2Vec resolves to the ExponentialWeight group ITS — the correct
// unbiased first hop.
template <bool IsDirected, bool Forward, RandomPickerType EdgePickerType>
DEVICE __forceinline__ void coop_pick_and_add_hop(
    const TemporalGraphView& view,
    WalkSetView              walk_set,
    const NodeDirPtrs&       ptrs,
    const size_t*            group_offsets,
    const int64_t*           first_ts,
    const size_t*            sorted_indices,
    const int64_t*           view_timestamps,
    const double*            s_cum_weights,
    const int                node_id,
    const size_t             node_group_begin,
    const size_t             node_group_end,
    const int                G,
    const size_t             node_edge_end,
    const size_t             walk_idx,
    const int                step_number,
    const int                max_walk_len,
    const int64_t            last_ts,
    const int64_t            cutoff,
    const double             r_group,
    const double             r_edge) {

    if constexpr (EdgePickerType == RandomPickerType::TemporalNode2Vec) {
        const size_t offset = walk_idx * static_cast<size_t>(max_walk_len)
                              + static_cast<size_t>(step_number);
        const int prev_node = step_number > 0
            ? walk_set.nodes[offset - 1]
            : -1;

        if (prev_node != -1) {
            int local_begin = 0;
            int local_end   = G;
            temporal_graph::filter_valid_groups_by_timestamp_slice<Forward>(
                group_offsets, first_ts, sorted_indices, view_timestamps,
                G, last_ts, cutoff, local_begin, local_end);
            if (local_begin >= local_end) return;

            const size_t valid_begin =
                node_group_begin + static_cast<size_t>(local_begin);
            const size_t valid_end =
                node_group_begin + static_cast<size_t>(local_end);

            const long edge_idx =
                temporal_graph::pick_random_temporal_node2vec_device<Forward, IsDirected>(
                    view, node_id, prev_node,
                    valid_begin, valid_end,
                    node_group_begin, node_group_end,
                    ptrs.node_ts_groups_offsets,
                    ptrs.node_ts_sorted_indices,
                    ptrs.weights, ptrs.weights_size,
                    r_group, r_edge);
            if (edge_idx == -1) return;

            add_edge_hop<IsDirected, Forward>(
                view, walk_set, node_id, edge_idx, walk_idx);
            return;
        }
        // prev_node == -1: fall through to the plain (ExponentialWeight) proposal.
    }

    const long local_pos =
        temporal_graph::find_group_pos_slice<Forward, EdgePickerType>(
            group_offsets, first_ts, sorted_indices, view_timestamps,
            ptrs.weights, ptrs.weights_size,
            node_group_begin, G, last_ts, cutoff, r_group, s_cum_weights);
    if (local_pos == -1) return;

    sample_edge_and_add_hop<IsDirected, Forward>(
        view, walk_set, ptrs,
        group_offsets, local_pos, G, node_edge_end, node_id,
        walk_idx, r_edge);
}

#endif // HAS_CUDA

} // namespace tempest

#endif // NODE_GROUPED_KERNELS_COMMON_CUH
