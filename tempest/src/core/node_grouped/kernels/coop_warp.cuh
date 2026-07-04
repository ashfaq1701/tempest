#ifndef NODE_GROUPED_KERNELS_COOP_WARP_CUH
#define NODE_GROUPED_KERNELS_COOP_WARP_CUH

#include "common.cuh"
#include "per_walk.cuh"
#include "../../../common/picker_dispatch.cuh"
#include "../../../common/cuda_config.cuh"

namespace tempest {

#ifdef HAS_CUDA

// __syncwarp() only — block-wide sync would deadlock idle warps in last block
template <bool IsDirected, bool Forward, RandomPickerType EdgePickerType>
__global__ void node_grouped_warp_smem_kernel(
    TemporalGraphView view,
    WalkSetView walk_set,
    const int* __restrict__ sorted_walk_idx,
    const int* __restrict__ node_walk_nodes,
    const int* __restrict__ node_walk_starts,
    const int* __restrict__ node_walk_counts,
    const int* __restrict__ num_tasks_ptr,
    const int step_number,
    const int max_walk_len,
    const uint64_t base_seed) {

    if (step_number >= max_walk_len - 1) return;

    const int warp_id = static_cast<int>(threadIdx.x >> 5);
    const int lane_id = static_cast<int>(threadIdx.x & 31u);
    const int task_id =
        static_cast<int>(blockIdx.x * (blockDim.x >> 5)) + warp_id;

    if (task_id >= *num_tasks_ptr) return;

    const auto ptrs = resolve_node_dir_ptrs<IsDirected, Forward>(view);

    const int    node_id          = node_walk_nodes[task_id];
    const int    walk_start       = node_walk_starts[task_id];
    const int    walk_count       = node_walk_counts[task_id];
    const size_t node_group_begin = ptrs.count_ts_group_per_node[node_id];
    const size_t node_group_end   = ptrs.count_ts_group_per_node[node_id + 1];
    const int    G                = static_cast<int>(node_group_end - node_group_begin);
    const size_t node_edge_end    = ptrs.node_edge_offsets[node_id + 1];

    constexpr int  kGCap     = coop_warp_smem_g_cap<EdgePickerType>();
    constexpr bool kIsWeight =
        !random_pickers::is_index_based_picker(EdgePickerType);
    constexpr size_t kHeaderBytes        = 64;
    constexpr size_t kGroupOffsetsOffset = kHeaderBytes;
    constexpr size_t kFirstTsOffset      =
        kGroupOffsetsOffset + sizeof(size_t) * kGCap;
    constexpr size_t kCumWeightsOffset   =
        kFirstTsOffset + sizeof(int64_t) * kGCap;
    constexpr size_t kPerWarpBytes       = kIsWeight
        ? kCumWeightsOffset + sizeof(double) * kGCap
        : kFirstTsOffset    + sizeof(int64_t) * kGCap;

    extern __shared__ unsigned char s_pool[];
    unsigned char* const s_my =
        s_pool + static_cast<size_t>(warp_id) * kPerWarpBytes;
    size_t* const  s_group_offsets =
        reinterpret_cast<size_t*>(s_my + kGroupOffsetsOffset);
    int64_t* const s_first_ts      =
        reinterpret_cast<int64_t*>(s_my + kFirstTsOffset);
    double* const  s_cum_weights   = kIsWeight
        ? reinterpret_cast<double*>(s_my + kCumWeightsOffset)
        : nullptr;

    // preload kills the 3-deep dependent load chain at search time
    for (int p = lane_id; p < G; p += 32) {
        const size_t ts_group_offset =
            ptrs.node_ts_groups_offsets[node_group_begin + p];
        s_group_offsets[p] = ts_group_offset;
        s_first_ts[p] =
            view.timestamps[ptrs.node_ts_sorted_indices[ts_group_offset]];
        if constexpr (kIsWeight) {
            s_cum_weights[p] = ptrs.weights[node_group_begin + p];
        }
    }
    __syncwarp();

    for (int k = lane_id; k < walk_count; k += 32) {
        const size_t walk_idx =
            static_cast<size_t>(sorted_walk_idx[walk_start + k]);
        const size_t offset   =
            walk_idx * static_cast<size_t>(max_walk_len)
            + static_cast<size_t>(step_number);
        const int64_t last_ts = walk_set.timestamps[offset];
        const int64_t cutoff  = walk_set.cutoffs[walk_idx];

        PhiloxState rng;
        init_philox_state(rng, base_seed, walk_idx,
                          step_kernel_philox_offset(step_number));
        const double r_group = draw_u01_philox(rng);
        const double r_edge  = draw_u01_philox(rng);

        coop_pick_and_add_hop<IsDirected, Forward, EdgePickerType>(
            view, walk_set, ptrs,
            GroupSource::smem(s_group_offsets, s_first_ts, s_cum_weights),
            node_id, node_group_begin, node_group_end, G, node_edge_end,
            walk_idx, step_number, max_walk_len, last_ts, cutoff,
            r_group, r_edge);
    }
}

template <RandomPickerType PickerType>
HOST inline size_t warp_smem_dynamic_smem_bytes(const dim3& block_dim) {
    constexpr size_t kGCap     = static_cast<size_t>(
        coop_warp_smem_g_cap<PickerType>());
    constexpr bool   kIsWeight =
        !random_pickers::is_index_based_picker(PickerType);
    constexpr size_t kPerGroupBytes =
        sizeof(size_t) + sizeof(int64_t) + (kIsWeight ? sizeof(double) : 0);
    constexpr size_t kPerWarp =
        size_t{64} + kPerGroupBytes * kGCap;
    return kPerWarp * static_cast<size_t>(block_dim.x >> 5);
}

template <bool IsDirected, bool Forward>
inline void dispatch_node_grouped_warp_smem_kernel(
    const TemporalGraphView& view,
    WalkSetView walk_set_view,
    const int* sorted_walk_idx,
    const int* node_walk_nodes,
    const int* node_walk_starts,
    const int* node_walk_counts,
    const int* num_tasks_ptr,
    const int step_number,
    const int max_walk_len,
    const RandomPickerType edge_picker_type,
    const uint64_t base_seed,
    const dim3& grid,
    const dim3& block_dim,
    const cudaStream_t stream) {

    dispatch_picker_type(edge_picker_type, [&](auto edge_tag) {
        constexpr auto kEdge = decltype(edge_tag)::value;
        const size_t smem_bytes = warp_smem_dynamic_smem_bytes<kEdge>(block_dim);
        node_grouped_warp_smem_kernel<IsDirected, Forward, kEdge>
            <<<grid, block_dim, smem_bytes, stream>>>(
                view, walk_set_view,
                sorted_walk_idx,
                node_walk_nodes, node_walk_starts, node_walk_counts,
                num_tasks_ptr,
                step_number, max_walk_len, base_seed);
    });
}

// no panel; coop here buys L1/L2 residency across lanes
template <bool IsDirected, bool Forward, RandomPickerType EdgePickerType>
__global__ void node_grouped_warp_global_kernel(
    TemporalGraphView view,
    WalkSetView walk_set,
    const int* __restrict__ sorted_walk_idx,
    const int* __restrict__ node_walk_nodes,
    const int* __restrict__ node_walk_starts,
    const int* __restrict__ node_walk_counts,
    const int* __restrict__ num_tasks_ptr,
    const int step_number,
    const int max_walk_len,
    const uint64_t base_seed) {

    if (step_number >= max_walk_len - 1) return;

    const int warp_id = static_cast<int>(threadIdx.x >> 5);
    const int lane_id = static_cast<int>(threadIdx.x & 31u);
    const int task_id =
        static_cast<int>(blockIdx.x * (blockDim.x >> 5)) + warp_id;

    if (task_id >= *num_tasks_ptr) return;

    const auto ptrs = resolve_node_dir_ptrs<IsDirected, Forward>(view);

    const int    node_id          = node_walk_nodes[task_id];
    const int    walk_start       = node_walk_starts[task_id];
    const int    walk_count       = node_walk_counts[task_id];
    const size_t node_group_begin = ptrs.count_ts_group_per_node[node_id];
    const size_t node_group_end   = ptrs.count_ts_group_per_node[node_id + 1];
    const int    G                = static_cast<int>(node_group_end - node_group_begin);
    const size_t node_edge_end    = ptrs.node_edge_offsets[node_id + 1];

    const size_t* group_offsets_slice =
        ptrs.node_ts_groups_offsets + node_group_begin;

    for (int k = lane_id; k < walk_count; k += 32) {
        const size_t walk_idx =
            static_cast<size_t>(sorted_walk_idx[walk_start + k]);
        const size_t offset   =
            walk_idx * static_cast<size_t>(max_walk_len)
            + static_cast<size_t>(step_number);
        const int64_t last_ts = walk_set.timestamps[offset];
        const int64_t cutoff  = walk_set.cutoffs[walk_idx];

        PhiloxState rng;
        init_philox_state(rng, base_seed, walk_idx,
                          step_kernel_philox_offset(step_number));
        const double r_group = draw_u01_philox(rng);
        const double r_edge  = draw_u01_philox(rng);

        coop_pick_and_add_hop<IsDirected, Forward, EdgePickerType>(
            view, walk_set, ptrs,
            GroupSource::global(group_offsets_slice,
                                ptrs.node_ts_sorted_indices, view.timestamps),
            node_id, node_group_begin, node_group_end, G, node_edge_end,
            walk_idx, step_number, max_walk_len, last_ts, cutoff,
            r_group, r_edge);
    }
}

template <bool IsDirected, bool Forward>
inline void dispatch_node_grouped_warp_global_kernel(
    const TemporalGraphView& view,
    WalkSetView walk_set_view,
    const int* sorted_walk_idx,
    const int* node_walk_nodes,
    const int* node_walk_starts,
    const int* node_walk_counts,
    const int* num_tasks_ptr,
    const int step_number,
    const int max_walk_len,
    const RandomPickerType edge_picker_type,
    const uint64_t base_seed,
    const dim3& grid,
    const dim3& block_dim,
    const cudaStream_t stream) {

    dispatch_picker_type(edge_picker_type, [&](auto edge_tag) {
        constexpr auto kEdge = decltype(edge_tag)::value;
        node_grouped_warp_global_kernel<IsDirected, Forward, kEdge>
            <<<grid, block_dim, 0, stream>>>(
                view, walk_set_view,
                sorted_walk_idx,
                node_walk_nodes, node_walk_starts, node_walk_counts,
                num_tasks_ptr,
                step_number, max_walk_len, base_seed);
    });
}

#endif // HAS_CUDA

} // namespace tempest

#endif // NODE_GROUPED_KERNELS_COOP_WARP_CUH
