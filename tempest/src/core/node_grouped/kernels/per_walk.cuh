#ifndef NODE_GROUPED_KERNELS_PER_WALK_CUH
#define NODE_GROUPED_KERNELS_PER_WALK_CUH

#include "common.cuh"
#include "../../../data/walk_set/walk_set_view.cuh"
#include "../../../data/temporal_graph_view.cuh"
#include "../../../graph/temporal_graph.cuh"
#include "../../../graph/edge_selectors.cuh"
#include "../../../utils/random.cuh"
#include "../../../utils/utils.cuh"
#include "../../../common/picker_dispatch.cuh"
#include "../../helpers.cuh"

namespace tempest {

#ifdef HAS_CUDA

// step 0 reserves [0,3) of Philox substream so all tiers stay bit-exact
DEVICE __forceinline__ uint64_t step_kernel_philox_offset(const int step_number) {
    if (step_number == 0) return 0ULL;
    constexpr uint64_t START_KERNEL_BUDGET = 3ULL;
    return START_KERNEL_BUDGET + static_cast<uint64_t>(step_number) * 2ULL;
}

template <bool IsDirected, bool Forward, RandomPickerType StartPickerType, bool Constrained>
__global__ void pick_start_edges_kernel(
    TemporalGraphView view,
    WalkSetView walk_set,
    const int* __restrict__ start_node_ids,
    const int max_walk_len,
    const size_t num_walks,
    const uint64_t base_seed) {

    const size_t walk_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (walk_idx >= num_walks) return;
    if (max_walk_len == 0) return;

    PhiloxState rng;
    init_philox_state(rng, base_seed, static_cast<uint64_t>(walk_idx));
    const double r_start_a = draw_u01_philox(rng);
    const double r_start_b = draw_u01_philox(rng);

    InternalEdge start_edge;
    if constexpr (Constrained) {
        const int64_t cutoff = walk_set.cutoffs[walk_idx];
        start_edge = temporal_graph::get_node_edge_at_device<Forward, StartPickerType, IsDirected>(
            view, start_node_ids[walk_idx], /*timestamp=*/-1, cutoff, /*prev=*/-1,
            r_start_a, r_start_b);
    } else {
        start_edge = temporal_graph::get_edge_at_device<Forward, StartPickerType>(
            view, /*timestamp=*/-1, r_start_a, r_start_b);
    }

    // dead start: walk stays len 0, filter drops it
    if (start_edge.i == -1) return;

    const int64_t sentinel_timestamp = Forward ? INT64_MIN : INT64_MAX;
    const int     start_src = start_edge.u;
    const int     start_dst = start_edge.i;
    const int64_t start_ts  = start_edge.ts;

    if constexpr (IsDirected) {
        if constexpr (Forward) {
            walk_set.add_hop(walk_idx, start_src, sentinel_timestamp);
            walk_set.add_hop(walk_idx, start_dst, start_ts, start_edge.edge_id);
        } else {
            walk_set.add_hop(walk_idx, start_dst, sentinel_timestamp);
            walk_set.add_hop(walk_idx, start_src, start_ts, start_edge.edge_id);
        }
    } else {
        // 3rd draw matches START_KERNEL_BUDGET=3 (bit-exact across tiers)
        int picked_node;
        if constexpr (Constrained) {
            picked_node = start_node_ids[walk_idx];
        } else {
            picked_node = pick_random_number(start_src, start_dst, draw_u01_philox(rng));
        }
        const int other_node = pick_other_number(start_src, start_dst, picked_node);
        walk_set.add_hop(walk_idx, picked_node, sentinel_timestamp);
        walk_set.add_hop(walk_idx, other_node, start_ts, start_edge.edge_id);
    }
}

template <bool IsDirected, bool Forward>
inline void dispatch_start_edges_kernel(
    const TemporalGraphView& view,
    WalkSetView walk_set_view,
    const int* start_node_ids,
    const bool constrained,
    const int max_walk_len,
    const size_t num_walks,
    const RandomPickerType start_picker_type,
    const uint64_t base_seed,
    const dim3& grid,
    const dim3& block_dim,
    const cudaStream_t stream) {

    dispatch_picker_type(start_picker_type, [&](auto start_tag) {
        constexpr auto kStart = decltype(start_tag)::value;
        dispatch_bool(constrained, [&](auto c_tag) {
            constexpr bool kC = decltype(c_tag)::value;
            pick_start_edges_kernel<IsDirected, Forward, kStart, kC>
                <<<grid, block_dim, 0, stream>>>(
                    view, walk_set_view, start_node_ids,
                    max_walk_len, num_walks, base_seed);
        });
    });
}

// writes slot 0 only if start node has edges in walk direction (else dead-start)
template <bool Forward, bool IsDirected>
__global__ void prepopulate_start_slot_kernel(
    WalkSetView walk_set,
    TemporalGraphView view,
    const int* __restrict__ start_node_ids,
    const size_t num_walks) {

    const size_t walk_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (walk_idx >= num_walks) return;

    const int node_id = start_node_ids[walk_idx];
    if (!temporal_graph::is_node_active(view, node_id)) return;

    const size_t* count_ts_group_per_node =
        count_ts_group_per_node_for_dir<IsDirected, Forward>(view);
    const size_t group_begin = count_ts_group_per_node[node_id];
    const size_t group_end   = count_ts_group_per_node[node_id + 1];
    if (group_begin == group_end) return;  // dead start

    // Cutoff-aware dead start: groups are sorted ascending by time, so the
    // earliest edge lives in the first group. If even it is at/after the
    // cutoff, no edge qualifies and step 0 would find nothing — skip slot 0
    // so the walk stays length 0, matching the CPU / full-walk paths.
    const int64_t cutoff = walk_set.cutoffs[walk_idx];
    if (cutoff != NO_WALK_CUTOFF) {
        const auto ptrs = resolve_node_dir_ptrs<IsDirected, Forward>(view);
        const size_t first_edge = ptrs.node_ts_groups_offsets[group_begin];
        const int64_t earliest_ts =
            view.timestamps[ptrs.node_ts_sorted_indices[first_edge]];
        if (earliest_ts >= cutoff) return;
    }

    constexpr int64_t sentinel_timestamp = Forward ? INT64_MIN : INT64_MAX;
    walk_set.add_hop(walk_idx, node_id, sentinel_timestamp);
}

template <bool Forward, bool IsDirected>
inline void dispatch_prepopulate_start_slot_kernel(
    const TemporalGraphView& view,
    WalkSetView walk_set_view,
    const int* start_node_ids,
    const size_t num_walks,
    const dim3& grid,
    const dim3& block_dim,
    const cudaStream_t stream) {

    prepopulate_start_slot_kernel<Forward, IsDirected>
        <<<grid, block_dim, 0, stream>>>(
            walk_set_view, view, start_node_ids, num_walks);
}

// outcome depends only on (walk_idx, step_number, base_seed) — tier-independent
template <bool IsDirected, bool Forward, RandomPickerType EdgePickerType>
DEVICE __forceinline__ void advance_one_walk(
    TemporalGraphView view,
    WalkSetView walk_set,
    const int walk_idx_int,
    const int step_number,
    const int max_walk_len,
    const uint64_t base_seed) {

    if (walk_idx_int < 0) return;
    if (step_number >= max_walk_len - 1) return;

    const size_t walk_idx = static_cast<size_t>(walk_idx_int);
    const size_t offset = walk_idx * static_cast<size_t>(max_walk_len)
                          + static_cast<size_t>(step_number);

    const int     last_node = walk_set.nodes[offset];
    const int64_t last_ts   = walk_set.timestamps[offset];
    const int64_t cutoff    = walk_set.cutoffs[walk_idx];
    const int     prev_node = step_number > 0 ? walk_set.nodes[offset - 1] : -1;

    PhiloxState rng;
    init_philox_state(rng, base_seed, static_cast<uint64_t>(walk_idx),
                      step_kernel_philox_offset(step_number));
    const double r_a = draw_u01_philox(rng);
    const double r_b = draw_u01_philox(rng);

    const InternalEdge next_edge =
        temporal_graph::get_node_edge_at_device<Forward, EdgePickerType, IsDirected>(
            view, last_node, last_ts, cutoff, prev_node, r_a, r_b);

    if (next_edge.ts == -1) return;

    if constexpr (IsDirected) {
        walk_set.add_hop(walk_idx,
                         Forward ? next_edge.i : next_edge.u,
                         next_edge.ts, next_edge.edge_id);
    } else {
        const int next_node = pick_other_number(next_edge.u, next_edge.i, last_node);
        walk_set.add_hop(walk_idx, next_node, next_edge.ts, next_edge.edge_id);
    }
}

template <bool IsDirected, bool Forward, RandomPickerType EdgePickerType>
__global__ void node_grouped_solo_kernel(
    TemporalGraphView view,
    WalkSetView walk_set,
    const int* __restrict__ solo_walks,
    const int* __restrict__ num_solo_walks_ptr,
    const int step_number,
    const int max_walk_len,
    const uint64_t base_seed) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *num_solo_walks_ptr) return;

    advance_one_walk<IsDirected, Forward, EdgePickerType>(
        view, walk_set, solo_walks[i],
        step_number, max_walk_len, base_seed);
}

template <bool IsDirected, bool Forward>
inline void dispatch_node_grouped_solo_kernel(
    const TemporalGraphView& view,
    WalkSetView walk_set_view,
    const int* solo_walks,
    const int* num_solo_walks_ptr,
    const int step_number,
    const int max_walk_len,
    const RandomPickerType edge_picker_type,
    const uint64_t base_seed,
    const dim3& grid,
    const dim3& block_dim,
    const cudaStream_t stream) {

    dispatch_picker_type(edge_picker_type, [&](auto edge_tag) {
        constexpr auto kEdge = decltype(edge_tag)::value;
        node_grouped_solo_kernel<IsDirected, Forward, kEdge>
            <<<grid, block_dim, 0, stream>>>(
                view, walk_set_view,
                solo_walks, num_solo_walks_ptr,
                step_number, max_walk_len, base_seed);
    });
}

__global__ static void reverse_walks_kernel(
    WalkSetView walk_set, const size_t num_walks) {
    const size_t walk_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (walk_idx >= num_walks) return;
    walk_set.reverse_walk(walk_idx);
}

#endif // HAS_CUDA

} // namespace tempest

#endif // NODE_GROUPED_KERNELS_PER_WALK_CUH
